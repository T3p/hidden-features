import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .nnlinucb import NNLinUCB

@dataclass
class NNLeader(NNLinUCB):

    weight_spectral: Optional[float]=-0.001
    weight_l2features: Optional[float]=0

    def __init__(self, env: Any, model: nn.Module, device: Optional[str] = "cpu", batch_size: Optional[int] = 256, max_updates: Optional[int] = 1, learning_rate: Optional[float] = 0.001, weight_decay: Optional[float] = 0, buffer_capacity: Optional[int] = 10000, seed: Optional[int] = 0, reset_model_at_train: Optional[bool] = True, update_every_n_steps: Optional[int] = 100, noise_std: float = 1, delta: Optional[float] = 0.01, ucb_regularizer: Optional[float] = 1, bonus_scale: Optional[float] = 1,
    weight_mse: Optional[float]=1.,
    weight_spectral: Optional[float]=-0.001,
    weight_l2features: Optional[float]=0
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, reset_model_at_train, update_every_n_steps, noise_std, delta, ucb_regularizer, bonus_scale)
        self.weight_mse = weight_mse
        self.weight_spectral = weight_spectral
        self.weight_l2features = weight_l2features

    def _train_loss(self, b_features, b_rewards):
        loss = 0
        N = b_features.shape[0]
        # MSE LOSS
        if not np.isclose(self.weight_mse,0):
            prediction = self.model(b_features)
            mse_loss = F.mse_loss(prediction, b_rewards)
            self.writer.add_scalar('mse_loss', self.weight_mse * mse_loss, self.batch_counter)
            loss = loss + self.weight_mse * mse_loss

        #DETERMINANT or LOG_MINEIG LOSS
        if not np.isclose(self.weight_spectral,0):
            phi = self.model.embedding(b_features)
            A = torch.sum(phi[...,None]*phi[:,None], axis=0) + 1e-3 * torch.eye(phi.shape[1])
            # det_loss = torch.logdet(A)
            spectral_loss = torch.log(torch.linalg.eigvalsh(A).min()/N)
            self.writer.add_scalar('spectral_loss', self.weight_spectral * spectral_loss, self.batch_counter)
            loss = loss + self.weight_spectral * spectral_loss

        # FEATURES NORM LOSS
        if not np.isclose(self.weight_l2features,0):
            l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
            # l2 reg on parameters can be done in the optimizer
            # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
            self.writer.add_scalar('l2feat_loss', self.weight_l2features * l2feat_loss, self.batch_counter)
            loss = loss + self.weight_l2features * l2feat_loss

        # TOTAL LOSS
        self.writer.add_scalar('loss', loss, self.batch_counter)
        return loss
    
