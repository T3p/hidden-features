import numpy as np
from typing import Optional, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from ..replaybuffer import SimpleBuffer
from ..nnmodel import initialize_weights
import time
import copy
from ... import TORCH_FLOAT
import wandb
from .core import IncBase
from ...envs import hlsutils


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + torch.dot(u.T, Au)
    A_inv -= torch.outer(Au, Au) / (den)
    return A_inv, den

class NNLinUCBInc(IncBase):
    
    def __init__(
        self,
        env: Any,
        model: nn.Module,
        device: Optional[str]="cpu",
        batch_size: Optional[int]=256,
        max_updates: Optional[int]=1,
        learning_rate: Optional[float]=0.001,
        weight_decay: Optional[float]=0,
        buffer_capacity: Optional[int]=10000,
        seed: Optional[int]=0,
        update_every: Optional[int] = 100,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.,
        use_tb: Optional[bool]=True,
        use_wandb: Optional[bool]=False
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, update_every, use_tb, use_wandb)
        self.noise_std = noise_std
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale

    def reset(self) -> None:
        super().reset()
        dim = self.model.embedding_dim
        self.b_vec = torch.zeros(dim).to(self.device)
        self.inv_A = torch.eye(dim).to(self.device) / self.ucb_regularizer
        self.A = torch.zeros_like(self.inv_A)
        self.theta = torch.zeros(dim).to(self.device)
        self.param_bound = 1
        self.features_bound = 1
    
    @torch.no_grad()
    def play_action(self, features: np.ndarray):
        assert features.shape[0] == self.env.action_space.n
        dim = self.target_model.embedding_dim
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2
                                                      *self.t/self.ucb_regularizer)/self.delta))\
               + self.param_bound * np.sqrt(self.ucb_regularizer)
        # beta = np.sqrt(np.log(self.t+1))

        # get features for each action and make it tensor
        xt = torch.tensor(features, dtype=TORCH_FLOAT).to(self.device)
        net_features = self.target_model.embedding(xt)
        #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
        bonus = ((net_features @ self.inv_A)*net_features).sum(axis=1)
        bonus = self.bonus_scale * beta * torch.sqrt(bonus)
        ucb = net_features @ self.theta + bonus
        action = torch.argmax(ucb).item()
        if self.use_tb:
            self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        if self.use_wandb:
            wandb.log({'bonus selected action': bonus[action].item()}, step=self.t)
        assert 0 <= action < self.env.action_space.n, ucb
        return action

    def _update_after_change_of_target(self):
        #################################################
        # Recompute design matrix and weight
        with torch.no_grad():
            # A = np.eye(dim) * self.ucb_regularizer
            dim = self.target_model.embedding_dim
            self.b_vec = torch.zeros(dim).to(self.device)
            self.inv_A = torch.eye(dim).to(self.device) / self.ucb_regularizer
            self.A = torch.zeros_like(self.inv_A)
            self.features_bound = 0
            features, rewards = self.buffer.get_all()
            features = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
            rewards = torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)

            phi = self.target_model.embedding(features)
            # features
            max_norm = torch.norm(phi, p=2, dim=1).max().cpu()
            self.features_bound = max(self.features_bound, max_norm)
            self.b_vec = self.b_vec + (phi * rewards).sum(dim=0)
            #SM
            for v in phi:
                self.inv_A = inv_sherman_morrison(v.ravel(), self.inv_A)[0]
                self.A += torch.outer(v.ravel(),v.ravel())
            self.theta = self.inv_A @ self.b_vec
            self.param_bound = torch.linalg.norm(self.theta, 2).item()
            self.writer.add_scalar('param_bound', self.param_bound, self.t)
            self.writer.add_scalar('features_bound', self.features_bound, self.t)
            # min_eig = torch.linalg.eigvalsh(self.A/(self.t+1)).min() / self.features_bound
            # self.writer.add_scalar('min_eig_empirical_design', min_eig, self.t)

            pred = phi @ self.theta
            mse_loss = F.mse_loss(pred.reshape(-1,1), rewards)
            if self.use_tb:
                self.writer.add_scalar('mse_linear', mse_loss.item(), self.t)
            if self.use_wandb:
                wandb.log({'mse_linear': mse_loss.item()}, step=self.t)
            # # debug metric
            if hasattr(self.env, 'feature_matrix'):
            #     xx = optimal_features(self.env.feature_matrix, self.env.rewards)
            #     assert len(xx.shape) == 2
            #     xt = torch.FloatTensor(xx).to(self.device)
            #     phi = self.model.embedding(xt).detach().cpu().numpy()
            #     norm_v=np.linalg.norm(phi, ord=2, axis=1).max()
            #     mineig = min_eig_outer(phi, False) / phi.shape[0]
            #     self.writer.add_scalar('min_eig_design_opt', mineig/norm_v, self.t)

                # compute misspecification error on all samples
                nc,na,nd = self.env.feature_matrix.shape
                U = self.env.feature_matrix.reshape(-1, nd)
                xt = torch.tensor(U, dtype=TORCH_FLOAT) 
                H = self.model.embedding(xt)
                newfeatures = H.reshape(nc, na, self.model.embedding_dim)
                newreward = newfeatures @ self.theta
                max_err = np.abs(self.env.rewards - newreward.cpu().detach().numpy()).max()
                self.writer.add_scalar('max miss-specification', max_err, self.t)

                # IS HLS
                newfeatures = newfeatures.cpu().detach().numpy()
                hls_rank = hlsutils.hls_rank(newfeatures, self.env.rewards)
                ishls = 1 if hlsutils.is_hls(newfeatures, self.env.rewards) else 0
                hls_lambda = hlsutils.hls_lambda(newfeatures, self.env.rewards)
                self.writer.add_scalar('hls_lambda', hls_lambda, self.t)
                self.writer.add_scalar('hls_rank', hls_rank, self.t)
                self.writer.add_scalar('hls?', ishls, self.t)
    
    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        exp = (features, reward)
        self.buffer.append(exp)
        #############################
        # estimate linear component on the embedding + UCB part
        with torch.no_grad():
            xt = torch.tensor(features.reshape(1,-1), dtype=TORCH_FLOAT).to(self.device)
            v = self.target_model.embedding(xt).squeeze()
            self.A += torch.outer(v.ravel(),v.ravel())
            self.b_vec = self.b_vec + v * reward
            self.inv_A, den = inv_sherman_morrison(v, self.inv_A)
            # self.A_logdet += np.log(den)
            self.theta = self.inv_A @ self.b_vec
