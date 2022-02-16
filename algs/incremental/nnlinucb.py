import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

from .templates import XBModule

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + torch.dot(u.T, Au)
    A_inv -= torch.outer(Au, Au) / (den)
    return A_inv, den

@dataclass
class NNLinUCB(XBModule):

    update_every_n_steps: Optional[int] = 100
    noise_std: float=1
    delta: Optional[float]=0.01
    ucb_regularizer: Optional[float]=1
    weight_mse: Optional[float]=1
    bonus_scale: Optional[float]=1.

    def reset(self):
        super().reset()
        dim = self.model.embedding_dim
        self.b_vec = torch.zeros(dim, dtype=torch.float)
        self.inv_A = torch.eye(dim, dtype=torch.float) / self.ucb_regularizer
        self.theta = torch.zeros(dim, dtype=torch.float)
        self.param_bound = 1
        self.features_bound = 1
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)

    def _train_loss(self, b_features, b_rewards):
        loss = 0
        # MSE LOSS
        if not np.isclose(self.weight_mse,0):
            prediction = self.model(b_features)
            mse_loss = F.mse_loss(prediction, b_rewards)
            self.writer.add_scalar('mse_loss', self.weight_mse * mse_loss, self.batch_counter)
            loss = loss + self.weight_mse * mse_loss 
        return loss
    
    @torch.no_grad()
    def play_action(self, features: np.ndarray):
        assert features.shape[0] == self.env.action_space.n
        dim = self.target_model.embedding_dim
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.ucb_regularizer)/self.delta)) + self.param_bound * np.sqrt(self.ucb_regularizer)

        # get features for each action and make it tensor
        xt = torch.FloatTensor(features).to(self.device)
        net_features = self.target_model.embedding(xt)
        #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
        bonus = ((net_features @ self.inv_A)*net_features).sum(axis=1)
        bonus = self.bonus_scale * beta * torch.sqrt(bonus)
        ucb = net_features @ self.theta + bonus
        action = torch.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        assert 0 <= action < self.env.action_space.n, ucb

        return action

    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        exp = (features, reward)
        self.buffer.append(exp)


        if self.t % self.update_every_n_steps == 0 and self.t > self.batch_size:
            # copy target network
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

            # recompute all the design matrix
            self._compute_design_matrix()
        else:
            # estimate linear component on the embedding + UCB part
            with torch.no_grad():
                xt = torch.FloatTensor(features.reshape(1,-1)).to(self.device)
                v = self.target_model.embedding(xt).ravel()
                self.features_bound = max(self.features_bound, torch.norm(v, p=2).item())
                self.writer.add_scalar('features_bound', self.features_bound, self.t)

                self.b_vec = self.b_vec + v * reward
                self.inv_A, den = inv_sherman_morrison(v, self.inv_A)
                # self.A_logdet += np.log(den)
                self.theta = self.inv_A @ self.b_vec
                self.param_bound = torch.linalg.norm(self.theta, 2).item()
                self.writer.add_scalar('param_bound', self.param_bound, self.t)
    
    def _compute_design_matrix(self):
        features, rewards = self.buffer.get_all()
        torch_dataset = torch.utils.data.TensorDataset(
                torch.tensor(features, dtype=torch.float, device=self.device),
                torch.tensor(rewards.reshape(-1,1), dtype=torch.float, device=self.device)
                )

        loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=self.batch_size, shuffle=True)
            
        with torch.no_grad():
            # A = np.eye(dim) * self.ucb_regularizer
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
            #     A = A + np.sum(phi[...,None]*phi[:,None], axis=0)
            # # strange issue with making operations directly in pytorch
            # self.inv_A = torch.tensor(np.linalg.inv(A), dtype=torch.float)
            self.theta = self.inv_A @ self.b_vec
