import imp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .xbdiscrete import XBTorchDiscrete, Experience

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + torch.dot(u.T, Au)
    A_inv -= torch.outer(Au, Au) / (den)
    return A_inv, den

@dataclass
class NNLinUCB(XBTorchDiscrete):

    noise_std: float=1
    delta: Optional[float]=0.01
    weight_mse: Optional[float]=1
    bonus_scale: Optional[float]=1.

    def __post_init__(self):
        self.b_vec = torch.zeros(self.net.embedding_dim)
        self.inv_A = torch.eye(self.net.embedding_dim) / self.weight_l2param
        self.theta = torch.zeros(self.net.embedding_dim)
        self.param_bound = 1
        self.features_bound = 1

    def _train_loss(self, b_context, b_actions, b_rewards):
        loss = 0
        # MSE LOSS
        if not np.isclose(self.weight_mse,0):
            prediction = self.net(b_context, b_actions)
            mse_loss = F.mse_loss(prediction, b_rewards)
            self.writer.add_scalar('mse_loss', mse_loss, self.batch_counter)
            loss = loss + self.weight_mse * mse_loss 
        return loss
    
    @torch.no_grad()
    def play_action(self, context: np.ndarray):
        dim = self.net.embedding_dim
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.weight_l2param)/self.delta)) + self.param_bound * np.sqrt(self.weight_l2param)

        # get features for each action and make it tensor
        na = self.env.action_space.n  
        tile_p = [na] + [1]*len(context.shape)
        contexts = torch.FloatTensor(np.tile(context, tile_p)).to(self.device)
        actions = np.arange(na).reshape(-1,1)
        if self.enc:
            actions = self.enc.transform(actions)
        actions = torch.FloatTensor(actions).to(self.device)

        net_features = self.net.embedding(contexts, actions)
        #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
        ucb = ((net_features @ self.inv_A)*net_features).sum(axis=1)
        ucb = torch.sqrt(ucb)
        ucb = net_features @ self.theta + self.bonus_scale * beta * ucb
        action = torch.argmax(ucb).item()
        assert 0 <= action < na, ucb

        return action

    def add_sample(self, context: np.ndarray, action: int, reward: float) -> None:
        if self.enc:
            action = self.enc.transform(np.array([action]).reshape(-1,1)).ravel()
        exp = Experience(context, action, reward)
        self.buffer.append(exp)

        # estimate linear component on the embedding + UCB part

        with torch.no_grad():
            x = torch.FloatTensor(context[np.newaxis, ...]).to(self.device)
            a = torch.FloatTensor(action.reshape(1,-1)).to(self.device)
            v = self.net.embedding(x, a).ravel()
            self.features_bound = max(self.features_bound, torch.norm(v, p=2).item())

            self.b_vec = self.b_vec + v * reward
            self.inv_A, den = inv_sherman_morrison(v, self.inv_A)
            # self.A_logdet += np.log(den)
            self.theta = self.inv_A @ self.b_vec
            self.param_bound = torch.linalg.norm(self.theta, 2).item()
    
    def _post_train(self, loader=None):
        with torch.no_grad():
            # A = np.eye(dim) * self.weight_l2param
            self.b_vec = torch.zeros(self.net.embedding_dim)
            self.inv_A = torch.eye(self.net.embedding_dim) / self.weight_l2param
            self.features_bound = 0
            for b_context, b_actions, b_rewards in loader:
                phi = self.net.embedding(b_context, b_actions) #.cpu().detach().numpy()

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
