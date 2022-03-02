import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.special import expit as sigmoid

from .templates import XBModule
from ..linear import inv_sherman_morrison


def sigmoid_nonlinearity(param_bound, features_bound):
    z = param_bound * features_bound
    return 2 * (np.cosh(z) + 1)

class NNLogisticUCB(XBModule):

    def __init__(self, 
        env: Any, 
        model: nn.Module, 
        device: Optional[str] = "cpu", 
        batch_size: Optional[int] = 256, 
        max_updates: Optional[int] = 1, 
        learning_rate: Optional[float] = 0.001, 
        weight_decay: Optional[float] = 0, 
        buffer_capacity: Optional[int] = 10000, 
        seed: Optional[int] = 0, 
        reset_model_at_train: Optional[bool] = True, 
        update_every_n_steps: Optional[int] = 100,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.,
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, reset_model_at_train, update_every_n_steps)
        self.np_random = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale

    def reset(self) -> None:
        super().reset()
        dim = self.model.embedding_dim
        self.b_vec = torch.zeros(dim, dtype=torch.float)
        self.inv_A = torch.eye(dim, dtype=torch.float) / self.ucb_regularizer
        self.theta = torch.zeros(dim, dtype=torch.float)
        self.param_bound = 1
        self.features_bound = 1

    def _train_loss(self, b_features, b_rewards):
        output = self.model(b_features)
        loss = F.binary_cross_entropy(output, b_rewards)
        return loss
    
    @torch.no_grad()
    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.env.action_space.n
        xt = torch.FloatTensor(features).to(self.device)
        net_features = self.model.embedding(xt)
        prediction = self.model(xt).squeeze()
        dim = net_features.shape[1]
        nonlinearity_coeff = sigmoid_nonlinearity(1. + self.param_bound, self.features_bound)
        beta = nonlinearity_coeff * self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2*self.t/self.ucb_regularizer)/self.delta)) + self.param_bound * np.sqrt(self.ucb_regularizer)

        # get features for each action and make it tensor
        bonus = ((net_features @ self.inv_A)*net_features).sum(axis=1)
        bonus = self.bonus_scale * beta * np.sqrt(bonus)
        ucb = prediction + bonus
        action = torch.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        assert 0 <= action < self.env.action_space.n, ucb
        return action
    
    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        exp = (features, reward)
        self.buffer.append(exp)

        # estimate linear component on the embedding + UCB part
        with torch.no_grad():
            xt = torch.FloatTensor(features.reshape(1,-1)).to(self.device)
            v = self.model.embedding(xt).squeeze()
            self.b_vec = self.b_vec + v * reward
            self.inv_A, den = inv_sherman_morrison(v, self.inv_A)
            # self.A_logdet += np.log(den)
            self.theta = self.inv_A @ self.b_vec
    
    def _post_train(self, loader=None):

        # recompute design matrix after update of neural network
        with torch.no_grad():
            dim = self.model.embedding_dim
            self.b_vec = torch.zeros(dim)
            self.inv_A = torch.eye(dim) / self.ucb_regularizer
            self.features_bound = 0
            for b_features, b_rewards in loader:
                phi = self.model.embedding(b_features) #.cpu().detach().numpy()

                # features
                max_norm = torch.norm(phi, p=2, dim=1).max()
                self.features_bound = max(self.features_bound, max_norm)
                self.b_vec = self.b_vec + (phi * b_rewards).sum(dim=0)
                #SM
                for v in phi:
                    self.inv_A = inv_sherman_morrison(v.ravel(), self.inv_A)[0]
        

        
