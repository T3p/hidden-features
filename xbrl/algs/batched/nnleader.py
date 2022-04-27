import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .nnlinucb import NNLinUCB, TORCH_FLOAT


class NNLeader(NNLinUCB):

    def __init__(
        self, env: Any, model: nn.Module, device: Optional[str] = "cpu", batch_size: Optional[int] = 256,
            max_updates: Optional[int] = 1, learning_rate: Optional[float] = 0.001,
            weight_decay: Optional[float] = 0, buffer_capacity: Optional[int] = 10000,
            seed: Optional[int] = 0, reset_model_at_train: Optional[bool] = True,
            update_every: Optional[int] = 100, noise_std: float = 1,
            delta: Optional[float] = 0.01, ucb_regularizer: Optional[float] = 1,
            weight_mse: Optional[float] = 1, bonus_scale: Optional[float] = 1,
            weight_spectral: Optional[float]=-0.001,
            weight_l2features: Optional[float]=0,
            weight_orth: Optional[float] = 0,
            weight_rayleigh: Optional[float] = 0,
            train_reweight: Optional[bool]=False
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, reset_model_at_train, update_every, noise_std, delta, ucb_regularizer, weight_mse, bonus_scale, train_reweight)
        self.weight_spectral = weight_spectral
        self.weight_l2features = weight_l2features
        self.weight_orth = weight_orth
        self.weight_rayleigh = weight_rayleigh
        if self.weight_rayleigh > 0:
            self.unit_vector = torch.ones(self.model.embedding_dim).to(self.device) / np.sqrt(self.model.embedding_dim)
            self.unit_vector.requires_grad = True
            self.unit_vector_optimizer = torch.optim.Adam([self.unit_vector], lr=self.learning_rate)


    def _train_loss(self, b_features, b_rewards, b_weights):
        prediction = self.model(b_features)
        mse_loss = F.mse_loss(prediction, b_rewards)
        self.writer.add_scalar('mse_loss', mse_loss.item(), self.batch_counter)

        phi = self.model.embedding(b_features)
        # nv=torch.norm(phi,p=2,dim=1).max().cpu().detach().numpy()
        A = torch.matmul(phi.transpose(1, 0), phi)
        HH = 0          
        for el in phi:
            HH += torch.outer(el,el)
        assert np.allclose(A.detach().numpy(), HH.detach().numpy()  )
        spectral_loss = torch.log(torch.linalg.eigvalsh(A).min() + 0.00001)
        self.writer.add_scalar('spectral_loss', spectral_loss, self.batch_counter)

        mse_weight = self.batch_counter / 200 
        # mse_weight = (self.tot_update) / (self.tot_update + 10)
        self.writer.add_scalar('mse_weight', mse_weight, self.batch_counter)

        loss = mse_weight * mse_loss - spectral_loss
        return loss, {}

    # def _train_loss(self, b_features, b_rewards, b_weights):
    #     loss = 0
    #     # MSE LOSS
    #     if not np.isclose(self.weight_mse,0):
    #         prediction = self.model(b_features)
    #         mse_loss = F.mse_loss(prediction, b_rewards)
    #         self.writer.add_scalar('mse_loss', self.weight_mse * mse_loss, self.batch_counter)
    #         loss = loss + self.weight_mse * mse_loss

    #     #DETERMINANT or LOG_MINEIG LOSS
    #     if not np.isclose(self.weight_spectral, 0):
    #         phi = self.model.embedding(b_features)
    #         # A = torch.sum(phi[...,None]*phi[:,None], axis=0) + 1e-3 * torch.eye(phi.shape[1]).to(self.device)
    #         A = torch.matmul(phi.transpose(1, 0), phi)
    #         # det_loss = torch.logdet(A)
    #         spectral_loss = torch.linalg.eigvalsh(A).min()
    #         # print(spectral_loss)
    #         self.writer.add_scalar('spectral_loss', self.weight_spectral * spectral_loss, self.batch_counter)
    #         loss = loss + self.weight_spectral * spectral_loss

    #     if not np.isclose(self.weight_orth, 0):
    #         batch_size = b_features.shape[0]
    #         b_phi = self.model.embedding(b_features)
    #         phi_1 = b_phi[: batch_size // 2]
    #         phi_2 = b_phi[batch_size // 2: ]

    #         phi_1_2 = torch.matmul(phi_1, phi_2.T)
    #         phi_1_1 = torch.einsum('sd, sd -> s', phi_1, phi_1)
    #         phi_2_2 = torch.einsum('sd, sd -> s', phi_2, phi_2)
    #         orth_loss = phi_1_2.pow(2).mean() - (phi_1_1.mean() + phi_2_2.mean())

    #         loss += self.weight_orth * orth_loss

    #     if not np.isclose(self.weight_rayleigh, 0):
    #         phi = self.model.embedding(b_features)
    #         A = torch.matmul(phi.T, phi) / phi.shape[0]
    #         # import pdb
    #         # pdb.set_trace()
    #         # compute loss to update the unit vector
    #         unit_vector_loss = torch.dot(self.unit_vector, torch.matmul(A.detach(), self.unit_vector))
    #         self.unit_vector_optimizer.zero_grad()
    #         unit_vector_loss.backward()
    #         self.unit_vector_optimizer.step()
    #         self.unit_vector.data = F.normalize(self.unit_vector.data, dim=0)
    #         # recompute the loss to update embedding
    #         phi = self.model.embedding(b_features)
    #         A = torch.matmul(phi.T, phi) / phi.shape[0]
    #         rayleigh_loss = - torch.dot(self.unit_vector.detach(), torch.matmul(A, self.unit_vector.detach()))
    #         loss += self.weight_rayleigh * rayleigh_loss




    #     # FEATURES NORM LOSS
    #     if not np.isclose(self.weight_l2features, 0):
    #         l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
    #         # l2 reg on parameters can be done in the optimizer
    #         # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
    #         self.writer.add_scalar('l2feat_loss', self.weight_l2features * l2feat_loss, self.batch_counter)
    #         loss = loss + self.weight_l2features * l2feat_loss

    #     # TOTAL LOSS
    #     self.writer.add_scalar('loss', loss, self.batch_counter)
    #     return loss
    
