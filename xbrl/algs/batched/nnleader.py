import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .nnlinucb import NNLinUCB


class NNLeader(NNLinUCB):

    def __init__(
        self, env: Any, model: nn.Module, device: Optional[str] = "cpu", batch_size: Optional[int] = 256, max_updates: Optional[int] = 1, learning_rate: Optional[float] = 0.001, weight_decay: Optional[float] = 0, buffer_capacity: Optional[int] = 10000, seed: Optional[int] = 0, reset_model_at_train: Optional[bool] = True, update_every_n_steps: Optional[int] = 100, noise_std: float = 1, delta: Optional[float] = 0.01, ucb_regularizer: Optional[float] = 1, weight_mse: Optional[float] = 1, bonus_scale: Optional[float] = 1,
        weight_spectral: Optional[float]=-0.001,
        weight_l2features: Optional[float]=0,
        weight_orth: Optional[float] = 0,
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, reset_model_at_train, update_every_n_steps, noise_std, delta, ucb_regularizer, weight_mse, bonus_scale)
        self.weight_spectral = weight_spectral
        self.weight_l2features = weight_l2features
        self.weight_orth = weight_orth


    def _train_loss(self, b_features, b_rewards):
        loss = 0
        # MSE LOSS
        if not np.isclose(self.weight_mse,0):
            prediction = self.model(b_features)
            mse_loss = F.mse_loss(prediction, b_rewards)
            self.writer.add_scalar('mse_loss', self.weight_mse * mse_loss, self.batch_counter)
            loss = loss + self.weight_mse * mse_loss

        #DETERMINANT or LOG_MINEIG LOSS
        if not np.isclose(self.weight_spectral, 0):
            phi = self.model.embedding(b_features)
            # A = torch.sum(phi[...,None]*phi[:,None], axis=0) + 1e-3 * torch.eye(phi.shape[1]).to(self.device)
            A = torch.matmul(phi.transpose(1, 0), phi)
            # det_loss = torch.logdet(A)
            spectral_loss = torch.linalg.eigvalsh(A).min()
            # print(spectral_loss)
            self.writer.add_scalar('spectral_loss', self.weight_spectral * spectral_loss, self.batch_counter)
            loss = loss + self.weight_spectral * spectral_loss

        if not np.isclose(self.weight_orth, 0):
            batch_size = b_features.shape[0]
            b_phi = self.model.embedding(b_features)
            phi_1 = b_phi[: batch_size // 2]
            phi_2 = b_phi[batch_size // 2: ]

            phi_1_2 = torch.matmul(phi_1, phi_2.T)
            phi_1_1 = torch.einsum('sd, sd -> s', phi_1, phi_1)
            phi_2_2 = torch.einsum('sd, sd -> s', phi_2, phi_2)
            orth_loss = phi_1_2.pow(2).mean() - (phi_1_1.mean() + phi_2_2.mean())

            loss += self.weight_orth * orth_loss


        # FEATURES NORM LOSS
        if not np.isclose(self.weight_l2features, 0):
            l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
            # l2 reg on parameters can be done in the optimizer
            # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
            self.writer.add_scalar('l2feat_loss', self.weight_l2features * l2feat_loss, self.batch_counter)
            loss = loss + self.weight_l2features * l2feat_loss

        # TOTAL LOSS
        self.writer.add_scalar('loss', loss, self.batch_counter)
        return loss
    
