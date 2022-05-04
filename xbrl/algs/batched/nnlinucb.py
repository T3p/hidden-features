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


class NNLinUCB(XBModule):

    def __init__(
        self, env: Any, model: nn.Module, cfg: DictConfig
    ) -> None:
        super().__init__(env, model, cfg)
        self.noise_std = cfg.noise_std
        self.delta = cfg.delta
        self.ucb_regularizer = cfg.ucb_regularizer
        self.bonus_scale = cfg.bonus_scale
        self.weight_mse = cfg.weight_mse
        self.device = cfg.device
        self.weight_spectral = cfg.weight_spectral
        self.weight_l2features = cfg.weight_l2features
        self.weight_orth = cfg.weight_orth
        self.weight_rayleigh = cfg.weight_rayleigh
        if self.weight_rayleigh > 0:
            self.unit_vector = torch.ones(self.model.embedding_dim).to(self.device) / np.sqrt(self.model.embedding_dim)
            self.unit_vector.requires_grad = True
            self.unit_vector_optimizer = torch.optim.Adam([self.unit_vector], lr=self.learning_rate)
        # initialization
        dim = self.model.embedding_dim
        self.b_vec = torch.zeros(dim, dtype=TORCH_FLOAT).to(self.device)
        self.inv_A = torch.eye(dim, dtype=TORCH_FLOAT).to(self.device) / self.ucb_regularizer
        self.A = torch.eye(dim, dtype=TORCH_FLOAT).to(self.device) * self.ucb_regularizer
        self.theta = torch.zeros(dim, dtype=TORCH_FLOAT).to(self.device)
        self.param_bound = np.sqrt(self.env.feature_dim)
        self.features_bound = np.sqrt(self.env.feature_dim)
        self.A_logdet = np.log(self.ucb_regularizer) * dim

    def _train_loss(self, b_features, b_rewards, b_weights):
        loss = 0
        metrics = {}
        # (weighted) MSE LOSS
        if not np.isclose(self.weight_mse,0):
            prediction = self.model(b_features)
            mse_loss = (b_weights * (prediction - b_rewards) ** 2).mean()
            self.writer.add_scalar('mse_loss', self.weight_mse * mse_loss, self.batch_counter)
            loss = loss + self.weight_mse * mse_loss

        #DETERMINANT or LOG_MINEIG LOSS
        if not np.isclose(self.weight_spectral, 0):
            phi = self.model.embedding(b_features)
            A = torch.matmul(phi.transpose(1, 0), phi) + self.ucb_regularizer * torch.eye(phi.shape[1])
            A /= phi.shape[0]
            # det_loss = torch.logdet(A)
            spectral_loss = - torch.linalg.eigvalsh(A).min()
            self.writer.add_scalar('spectral_loss', spectral_loss, self.batch_counter)
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

        if not np.isclose(self.weight_rayleigh, 0):
            phi = self.model.embedding(b_features)
            A = torch.matmul(phi.T, phi) / phi.shape[0]
            # compute loss to update the unit vector
            unit_vector_loss = torch.dot(self.unit_vector, torch.matmul(A.detach(), self.unit_vector))
            self.unit_vector_optimizer.zero_grad()
            unit_vector_loss.backward()
            self.unit_vector_optimizer.step()
            self.unit_vector.data = F.normalize(self.unit_vector.data, dim=0)
            # recompute the loss to update embedding
            phi = self.model.embedding(b_features)
            A = torch.matmul(phi.T, phi) / phi.shape[0]
            rayleigh_loss = - torch.dot(self.unit_vector.detach(), torch.matmul(A, self.unit_vector.detach()))
            loss += self.weight_rayleigh * rayleigh_loss

        # FEATURES NORM LOSS
        if not np.isclose(self.weight_l2features, 0):
            l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
            # l2 reg on parameters can be done in the optimizer
            # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
            self.writer.add_scalar('l2feat_loss', self.weight_l2features * l2feat_loss, self.batch_counter)
            loss = loss + self.weight_l2features * l2feat_loss

        # TOTAL LOSS
        self.writer.add_scalar('loss', loss.item(), self.batch_counter)
        # perform an SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        metrics['loss'] = loss.item()
        return metrics

    # def _train_loss(self, b_features, b_rewards, b_weights):
    #     loss = 0
    #     # MSE LOSS
    #     if not np.isclose(self.weight_mse,0):
    #         prediction = self.model(b_features)
    #         # mse_loss = F.mse_loss(prediction, b_rewards)
    #         # self.writer.add_scalar('mse_loss', self.weight_mse * mse_loss, self.batch_counter)
    #         mse_loss = (b_weights * (prediction - b_rewards)**2).mean()
    #         loss = loss + self.weight_mse * mse_loss
    #         with torch.no_grad():
    #             # debug metrics
    #             deb_mle = F.mse_loss(prediction, b_rewards)
    #
    #     return loss, {"mse_log": deb_mle.item()}

    @torch.no_grad()
    def play_action(self, features: np.ndarray):
        assert features.shape[0] == self.env.action_space.n
        dim = self.model.embedding_dim
        # beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2
        #                                               *self.t/self.ucb_regularizer)/self.delta))\
        #        + self.param_bound * np.sqrt(self.ucb_regularizer)
        val = self.A_logdet - dim * np.log(self.ucb_regularizer) - 2 * np.log(self.delta)
        beta = self.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.ucb_regularizer)
        # beta = np.sqrt(np.log(self.t+1))

        # get features for each action and make it tensor
        xt = torch.tensor(features, dtype=TORCH_FLOAT).to(self.device)
        net_features = self.model.embedding(xt)
        #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
        bonus = ((net_features @ self.inv_A)*net_features).sum(axis=1)
        bonus = self.bonus_scale * beta * torch.sqrt(bonus)
        ucb = net_features @ self.theta + bonus
        action = torch.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        assert 0 <= action < self.env.action_space.n, ucb

        if self.t % 100 == 0:
            self.logger.info(f"[{self.t}]bonus:\n {bonus}")
            self.logger.info(f"[{self.t}]prediction:\n {net_features @ self.theta}")
            self.logger.info(f"[{self.t}]ucb:\n {ucb}")
        return action

    def add_sample(self, features: np.ndarray, reward: float) -> None:
        exp = (features, reward)
        self.buffer.append(exp)

        # estimate linear component on the embedding + UCB part
        with torch.no_grad():
            xt = torch.tensor(features.reshape(1,-1), dtype=TORCH_FLOAT).to(self.device)
            v = self.model.embedding(xt).squeeze()

            self.A += torch.outer(v.ravel(),v.ravel())
            self.b_vec = self.b_vec + v * reward
            # # self.inv_A, den = inv_sherman_morrison(v, self.inv_A)
            # Au = self.inv_A @ v
            # den = 1 + torch.dot(v.T, Au)
            # self.inv_A -= torch.outer(Au, Au) / (den)
            # # self.A_logdet += np.log(den)
            # self.theta = self.inv_A @ self.b_vec
            self.theta = torch.linalg.solve(self.A, self.b_vec)
            self.inv_A = torch.linalg.inv(self.A)
            self.A_logdet = torch.logdet(self.A).cpu().item()

            if self.t % 50 == 0:
                f, r = self.buffer.get_all()
                torch_feat = torch.tensor(f, dtype=TORCH_FLOAT).to(self.device)
                torch_rew = torch.tensor(r.reshape(-1,1), dtype=TORCH_FLOAT).to(self.device)
                torch_phi = self.model.embedding(torch_feat)
                pred = torch_phi @ self.theta
                mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
                self.writer.add_scalar('mse_linear', mse_loss.item(), self.t)
                # print(f"{self.t} - mse: {mse_loss.item()}")

            if self.t % 100 == 0:

                nc,na,nd = self.env.feature_matrix.shape
                U = self.env.feature_matrix.reshape(-1, nd)
                xt = torch.tensor(U, dtype=TORCH_FLOAT).to(self.device)
                H = self.model.embedding(xt)
                newfeatures = H.reshape(nc, na, self.model.embedding_dim)
                newreward = newfeatures @ self.theta
                max_err = np.abs(self.env.rewards - newreward.cpu().detach().numpy()).max()
                self.writer.add_scalar('max miss-specification', max_err, self.t)


            self.features_bound = max(self.features_bound, torch.norm(v, p=2).cpu().item())
            self.writer.add_scalar('features_bound', self.features_bound, self.t)

            self.param_bound = torch.linalg.norm(self.theta, 2).cpu().item()
            self.writer.add_scalar('param_bound', self.param_bound, self.t)

    def _post_train(self, loader=None):
        with torch.no_grad():

            dim = self.model.embedding_dim
            f, r = self.buffer.get_all()
            # inv_A = torch.eye(dim) / self.ucb_regularizer
            torch_feat = torch.tensor(f, dtype=TORCH_FLOAT)
            torch_rew = torch.tensor(r.reshape(-1,1), dtype=TORCH_FLOAT)
            torch_phi = self.model.embedding(torch_feat)
            YYtt = (torch_phi.T @ torch_phi + self.ucb_regularizer *torch.eye(dim))
            BBtt = torch.sum(torch_phi * torch_rew, 0)
            thetatt = torch.linalg.solve(YYtt, BBtt)
            self.theta = thetatt
            self.A = YYtt
            self.b_vec = BBtt
            self.inv_A = torch.linalg.inv(self.A)
            self.A_logdet = torch.logdet(self.A)
            # for el in torch_phi:
            #     # el = el.squeeze()
            #     Au = inv_A @ el
            #     den = 1 + torch.dot(el.T, Au)
            #     inv_A -= torch.outer(Au, Au) / (den)
            # theta2 = inv_A @ BBtt

            # # numpy
            # np_f_32 = np.float32(torch_phi.numpy())
            # np_r_32 = np.float32(torch_rew.numpy())
            # np_inv_A_32 = np.float32(np.eye(dim)) / self.ucb_regularizer
            # np_b_32 = np.sum(np_f_32 * np_r_32, 0)
            # for el in np_f_32:
            #     Au = np_inv_A_32 @ el
            #     den = 1 + np.dot(el.T, Au)
            #     np_inv_A_32 -= np.outer(Au, Au) / (den)
            # np_theta_32 = np_inv_A_32 @ np_b_32



            # # A = np.eye(dim) * self.ucb_regularizer
            # self.b_vec = torch.zeros(dim).to(self.device)
            # self.inv_A = torch.eye(dim).to(self.device) / self.ucb_regularizer
            # self.A = torch.zeros_like(self.inv_A)
            # self.features_bound = 0
            # for b_features, b_rewards, b_weights in loader:
            #     phi = self.model.embedding(b_features) #.cpu().detach().numpy()

            #     # features
            #     max_norm = torch.norm(phi, p=2, dim=1).max().cpu()
            #     self.features_bound = max(self.features_bound, max_norm)
            #     self.b_vec = self.b_vec + (phi * b_rewards).sum(dim=0)
            #     #SM
            #     for v in phi:
            #         Au = self.inv_A @ v
            #         den = 1 + torch.dot(v.T, Au)
            #         self.inv_A -= torch.outer(Au, Au) / (den)
            #         self.A += torch.outer(v.ravel(),v.ravel())
            # #     A = A + np.sum(phi[...,None]*phi[:,None], axis=0)
            # # # strange issue with making operations directly in pytorch
            # # self.inv_A = torch.tensor(np.linalg.inv(A), dtype=torch.float)
            # self.theta = self.inv_A @ self.b_vec
            self.features_bound = torch.norm(torch_phi, p=2, dim=1).max().cpu().item()
            self.param_bound = torch.linalg.norm(self.theta, 2).item()
            self.writer.add_scalar('param_bound', self.param_bound, self.t)
            self.writer.add_scalar('features_bound', self.features_bound, self.t)
            # min_eig = torch.linalg.eigvalsh(self.A/(self.t+1)).min() / self.features_bound
            # self.writer.add_scalar('min_eig_empirical_design', min_eig, self.t)

            pred = torch_phi @ self.theta
            mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
            self.writer.add_scalar('mse_linear', mse_loss.item(), self.t)
            # print(f"{self.t}mse1 (used): ", mse_loss.item())

            # pred = torch_phi @ thetatt
            # mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
            # print(f"{self.t}mse2: ", mse_loss.item())

            # pred = torch_phi @ theta2
            # mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
            # print(f"{self.t}mse3: ", mse_loss.item())

            # jjj = torch.linalg.inv(YYtt)
            # pred = torch_phi @ (jjj @ BBtt)
            # mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
            # print(f"{self.t}mse4: ", mse_loss.item())

            # pred = np_f_32 @ np_theta_32
            # mse_loss = ((pred - np_r_32)**2).mean()
            # print(f"{self.t}mse5(np 32): ", mse_loss)


            # self.inv_A = jjj
            # self.b_vec = BBtt
            # self.theta = jjj @ BBtt

            # debug metric
            if hasattr(self.env, 'feature_matrix'):
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
                rank_phi = hlsutils.rank(newfeatures, None)
                self.writer.add_scalar('hls_lambda', hls_lambda, self.t)
                self.writer.add_scalar('hls_rank', hls_rank, self.t)
                self.writer.add_scalar('hls?', ishls, self.t)
                self.writer.add_scalar('rank_phi', rank_phi, self.t)




