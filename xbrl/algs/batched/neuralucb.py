import pdb

import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .templates import XBModule
from ...envs import hlsutils
from ... import TORCH_FLOAT
from omegaconf import DictConfig
import wandb
import os

class NeuralUCB(XBModule):

    def __init__(
        self, env: Any, cfg: DictConfig, model: Optional[nn.Module] = None
    ) -> None:
        super().__init__(env, cfg, model)
        self.noise_std = cfg.noise_std
        self.delta = cfg.delta
        self.bonus_scale = cfg.bonus_scale
        self.weight_mse = cfg.weight_mse
        self.device = cfg.device
        self.weight_l2features = cfg.weight_l2features
        self.adaptive_bonus_linucb = cfg.adaptive_bonus_linucb
        self.save_model_at_train = cfg.save_model_at_train
        
        #turn into input params:
        self.C1 = 1.
        self.C2 = 1.
        self.C3 = 1.
        self.gd_steps = 100
        self.param_bound = 1.
      
        #Network size
        with torch.no_grad():
            nn_weights = [p for p in self.model.parameters()]
            self.depth = (len(nn_weights) - 1) // 2
            self.width = len(nn_weights[-2])
            self.dim = dim = sum(torch.numel(W) for W in nn_weights) #tot number of network weights = size of gradient vector
        
        #Initialization
        self.b_vec = torch.zeros(dim, dtype=TORCH_FLOAT).to(self.device)
        self.inv_A = torch.eye(dim, dtype=TORCH_FLOAT).to(self.device) / self.weight_decay
        self.A = torch.eye(dim, dtype=TORCH_FLOAT).to(self.device) * self.weight_decay
        with torch.no_grad():
            self.theta = torch.ravel(nn_weights[-1]).clone().detach().to(self.device)
        self.param_bound = np.sqrt(self.env.feature_dim)
        self.features_bound = np.sqrt(self.env.feature_dim)
        self.A_logdet = np.log(self.weight_decay) * dim
        self.np_random = np.random.RandomState(self.seed)
        self.is_random_step = 1
        

    def _train_loss(self, exp_features_tensor, exp_rewards_tensor, b_features, b_rewards, b_weights, b_all_features):
        loss = 0
        metrics = {}
        # (weighted) MSE LOSS
        prediction = self.model(exp_features_tensor)
        mse_loss = F.mse_loss(prediction, exp_rewards_tensor)
        metrics['mse_loss'] = mse_loss.cpu().item()
        loss = loss + self.weight_mse * mse_loss
        #l2 regularization is done by the optimizer with self.weight_decay as reg. param. and self.learning_rate as step size

        # perform an SGD step
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
        self.optimizer.step()
        metrics['train_loss'] = loss.cpu().item()
        
        return metrics

    def play_action(self, features: np.ndarray):
        assert features.shape[0] == self.env.action_space.n

        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)

        #Extract gradient
        n_actions = self.env.action_space.n
        _gradients = []
        predicted_rewards = np.sqrt(self.width) * torch.ravel(self.model(features_tensor)) 
        for a in range(n_actions):
            f = predicted_rewards[a]
            self.model.zero_grad()
            f.backward(retain_graph=True)
            _gradients.append(torch.cat([p.grad.view(-1) for p in self.model.parameters()]))
        gradients = torch.stack(_gradients)
        dim = gradients[0].shape[0]
        
        # Compute bonus
        width_coeff = np.sqrt(np.log(self.width)) / self.width**(1/6)
        val_1 = 1. + self.C1 * width_coeff * self.depth**4 * self.t**(7/6) / self.weight_decay**(7/6)
        val_2 = self.A_logdet - dim * np.log(self.weight_decay) + self.C2 * width_coeff * self.depth**4 * self.t**(5/3) /self.weight_decay**(1/6) - 2*np.log(self.delta)
        val_3 = self.weight_decay + self.C3 * self.t * self.depth
        val_4 = ((1 - self.learning_rate * self.width * self.weight_decay)**(self.gd_steps/2) * 
                 np.sqrt(self.t/self.weight_decay) + width_coeff * self.depth**(7/2) 
                 * self.t**(5/3) / self.weight_decay**(5/3) * (1 + np.sqrt(self.t / self.weight_decay)))
        beta = np.sqrt(val_1) * (self.bonus_scale * np.sqrt(val_2) + np.sqrt(self.weight_decay) * self.param_bound) + val_3 * val_4
        
        bonus = beta * torch.sqrt((torch.matmul(gradients, self.inv_A) * gradients).sum(axis=1) / self.width)
        
        ucb = predicted_rewards + bonus
        action = torch.argmax(ucb).item()
        if self.use_tb:
            self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        if self.use_wandb:
            wandb.log({'bonus selected action': bonus[action].item()}, step=self.t)
        assert 0 <= action < self.env.action_space.n, ucb

        self.gradient = gradients[action]
        
        if self.t % 100 == 0:
            self.logger.info(f"[{self.t}]bonus:\n {bonus}")
            self.logger.info(f"[{self.t}]prediction:\n {predicted_rewards}")
            self.logger.info(f"[{self.t}]ucb:\n {ucb}")
    
        return action

    def add_sample(self, feature: np.ndarray, reward: float, all_features: np.ndarray) -> None:
        exp = (feature, reward, all_features, self.t, self.is_random_step)
        self.buffer.append(exp)
        if self.is_random_step:
            self.explorative_buffer.append((feature, reward))
        self.A += torch.outer(self.gradient, self.gradient) / self.width
        self.inv_A = torch.linalg.inv(self.A)
        self.A_logdet = torch.logdet(self.A).cpu().item()

        if self.use_tb:
            self.writer.add_scalar('features_bound', self.features_bound, self.t)
            self.writer.add_scalar('param_bound', self.param_bound, self.t)
        if self.use_wandb:
            wandb.log({'param_bound': self.param_bound}, step=self.t)


    def _post_train(self, loader=None) -> None:
        if self.model is None:
            return None
        batch_features, batch_rewards, _, _, _ = self.buffer.get_all()
        features_tensor = torch.tensor(batch_features, dtype=TORCH_FLOAT, device=self.device)
        rewards_tensor = torch.tensor(batch_rewards, dtype=TORCH_FLOAT, device=self.device)
        
        with torch.no_grad():
            nn_weights = [p for p in self.model.parameters()]
            self.theta = torch.ravel(nn_weights[-1]).clone().detach().to(self.device)

        prediction = self.model(features_tensor)
        mse_loss = (prediction - rewards_tensor).pow(2).mean()
        self.writer.add_scalar('mse_linear', mse_loss.item(), self.t)

        if self.save_model_at_train and self.model:
            path = os.path.join(self.log_path, f"model_state_dict_n{self.t}.pt")
            torch.save(self.model.state_dict(), path)


