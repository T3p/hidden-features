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


class NNLinUCB(XBModule):

    def __init__(
        self, env: Any, cfg: DictConfig, model: Optional[nn.Module] = None
    ) -> None:
        super().__init__(env, cfg, model)
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
        if self.model is not None:
            dim = self.model.embedding_dim
        else:
            dim = self.env.feature_dim
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
            loss = loss + self.weight_mse * mse_loss
            metrics['mse_loss'] = mse_loss.cpu().item()

        #DETERMINANT or LOG_MINEIG LOSS
        if not np.isclose(self.weight_spectral, 0):
            phi = self.model.embedding(b_features)
            A = torch.matmul(phi.T, phi) + self.ucb_regularizer * torch.eye(phi.shape[1], device=self.device)
            A /= phi.shape[0]
            # det_loss = torch.logdet(A)
            spectral_loss = - torch.log(torch.linalg.eigvalsh(A)[0])
            loss = loss + self.weight_spectral * spectral_loss
            metrics['spectral_loss'] = spectral_loss.cpu().item()

        if not np.isclose(self.weight_orth, 0):
            batch_size = b_features.shape[0]
            phi = self.model.embedding(b_features)
            phi_1 = phi[: batch_size // 2]
            phi_2 = phi[batch_size // 2: ]

            phi_1_2 = torch.matmul(phi_1, phi_2.T)
            phi_1_1 = torch.einsum('sd, sd -> s', phi_1, phi_1)
            phi_2_2 = torch.einsum('sd, sd -> s', phi_2, phi_2)
            orth_loss = phi_1_2.pow(2).mean() - (phi_1_1.mean() + phi_2_2.mean())

            loss += self.weight_orth * orth_loss
            metrics['orth_loss'] = orth_loss.cpu().item()

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
            metrics['rayleigh_loss'] = rayleigh_loss.cpu().item()

        # FEATURES NORM LOSS
        if not np.isclose(self.weight_l2features, 0):
            phi = self.model.embedding(b_features)
            l2feat_loss = torch.mean(torch.norm(phi, p=2, dim=1))
            # l2 reg on parameters can be done in the optimizer
            # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
            loss = loss + self.weight_l2features * l2feat_loss
            metrics['l2feat_loss'] = l2feat_loss.cpu().item()


        # perform an SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        metrics['train_loss'] = loss.cpu().item()
        return metrics

    def play_action(self, features: np.ndarray):
        assert features.shape[0] == self.env.action_space.n
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)

        if self.model is not None:
            dim = self.model.embedding_dim
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
        else:
            dim = self.env.feature_dim
            phi = features_tensor
        # beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2
        #                                               *self.t/self.ucb_regularizer)/self.delta))\
        #        + self.param_bound * np.sqrt(self.ucb_regularizer)
        val = self.A_logdet - dim * np.log(self.ucb_regularizer) - 2 * np.log(self.delta)
        beta = self.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.ucb_regularizer)
        # beta = np.sqrt(np.log(self.t+1))
        #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
        bonus = (torch.matmul(phi, self.inv_A) * phi).sum(axis=1)
        bonus = self.bonus_scale * beta * torch.sqrt(bonus)
        ucb = torch.matmul(phi, self.theta) + bonus
        action = torch.argmax(ucb).item()
        if self.use_tb:
            self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        if self.use_wandb:
            wandb.log({'bonus selected action': bonus[action].item()}, step=self.t)
        assert 0 <= action < self.env.action_space.n, ucb

        if self.t % 100 == 0:
            self.logger.info(f"[{self.t}]bonus:\n {bonus}")
            self.logger.info(f"[{self.t}]prediction:\n {torch.matmul(phi, self.theta)}")
            self.logger.info(f"[{self.t}]ucb:\n {ucb}")
        return action

    def add_sample(self, feature: np.ndarray, reward: float) -> None:
        exp = (feature, reward)
        self.buffer.append(exp)

        # estimate linear component on the embedding + UCB part
        feature_tensor = torch.tensor(feature.reshape(1,-1), dtype=TORCH_FLOAT).to(self.device)
        if self.model is not None:
            with torch.no_grad():
                phi = self.model.embedding(feature_tensor).squeeze()
        else:
            phi = feature_tensor.squeeze()

        self.A += torch.outer(phi, phi)
        self.b_vec += phi * reward
        # # self.inv_A, den = inv_sherman_morrison(v, self.inv_A)
        # Au = self.inv_A @ v
        # den = 1 + torch.dot(v.T, Au)
        # self.inv_A -= torch.outer(Au, Au) / (den)
        # # self.A_logdet += np.log(den)
        # self.theta = self.inv_A @ self.b_vec
        self.theta = torch.linalg.solve(self.A, self.b_vec)
        self.inv_A = torch.linalg.inv(self.A)
        self.A_logdet = torch.logdet(self.A).cpu().item()

        self.features_bound = max(self.features_bound, torch.norm(phi, p=2).cpu().item())
        self.param_bound = torch.linalg.norm(self.theta, 2).cpu().item()
        if self.use_tb:
            self.writer.add_scalar('features_bound', self.features_bound, self.t)
            self.writer.add_scalar('param_bound', self.param_bound, self.t)
        if self.use_wandb:
            wandb.log({'features_bound': self.features_bound,
                       'param_bound': self.param_bound}, step=self.t)

        # log in tensorboard
        if self.t % 50 == 0:
            batch_features, batch_rewards = self.buffer.get_all()
            error, _ = self.compute_linear_error(batch_features, batch_rewards)
            mse_loss = error.pow(2).mean()
            if self.use_tb:
                self.writer.add_scalar('mse_linear', mse_loss, self.t)
            if self.use_wandb:
                wandb.log({'mse_linear': mse_loss}, step=self.t)

        if self.t % 100 == 0:
            if hasattr(self.env, 'feature_matrix'):
                num_context, num_action, dim = self.env.feature_matrix.shape
                all_features = self.env.feature_matrix.reshape(-1, dim)
                all_rewards = self.env.rewards.reshape(-1)
                error, _ = self.compute_linear_error(all_features, all_rewards)
                max_err = torch.abs(error).max()
                mean_abs_err = torch.abs(error).mean()
                if self.use_tb:
                    self.writer.add_scalar('max miss-specification', max_err.cpu().item(), self.t)
                    self.writer.add_scalar('mean abs miss-specification', mean_abs_err.cpu().item(), self.t)

                if self.use_wandb:
                    wandb.log({'max miss-specification': max_err.cpu().item()}, step=self.t)
                    wandb.log({'mean abs miss-specification': mean_abs_err.cpu().item()}, step=self.t)

    def compute_linear_error(self, features: np.ndarray, reward: np.ndarray):
        assert len(features.shape) == 2 and len(reward.shape) == 1
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
        rewards_tensor = torch.tensor(reward, dtype=TORCH_FLOAT).to(self.device)
        if self.model is not None:
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
        else:
            phi = features_tensor
        prediction = torch.matmul(phi, self.theta)
        error = prediction - rewards_tensor
        return error, phi


    def _post_train(self, loader=None) -> None:
        if self.model is None:
            return None
        dim = self.model.embedding_dim
        batch_features, batch_rewards = self.buffer.get_all()
        features_tensor = torch.tensor(batch_features, dtype=TORCH_FLOAT, device=self.device)
        rewards_tensor = torch.tensor(batch_rewards, dtype=TORCH_FLOAT, device=self.device)
        with torch.no_grad():
            phi = self.model.embedding(features_tensor)
        A = torch.matmul(phi.T, phi) + self.ucb_regularizer * torch.eye(dim, device=self.device)
        b_vec = torch.matmul(phi.T, rewards_tensor)
        theta = torch.linalg.solve(A, b_vec)
        assert torch.allclose(torch.matmul(A, theta), b_vec)
        self.theta = theta
        self.A = A
        self.b_vec = b_vec
        self.inv_A = torch.linalg.inv(self.A)
        self.A_logdet = torch.logdet(self.A).cpu().item()

        self.features_bound = torch.norm(phi, p=2, dim=1).max().cpu().item()
        self.param_bound = torch.linalg.norm(self.theta, 2).item()
        self.writer.add_scalar('param_bound', self.param_bound, self.t)
        self.writer.add_scalar('features_bound', self.features_bound, self.t)

        prediction = torch.matmul(phi, self.theta)
        mse_loss = (prediction - rewards_tensor).pow(2).mean()
        self.writer.add_scalar('mse_linear', mse_loss.item(), self.t)

        # debug metric
        if hasattr(self.env, 'feature_matrix'):
            num_context, num_action, dim = self.env.feature_matrix.shape
            all_features = self.env.feature_matrix.reshape(-1, dim)
            all_rewards = self.env.rewards.reshape(-1)
            error, phi = self.compute_linear_error(all_features, all_rewards)
            max_err = torch.abs(error).max()


            # IS HLS
            new_phi = phi.reshape(num_context, num_action, self.model.embedding_dim)
            new_phi = new_phi.cpu().numpy()
            hls_rank = hlsutils.hls_rank(new_phi, self.env.rewards, tol=1e-4)
            ishls = 1 if hlsutils.is_hls(new_phi, self.env.rewards, tol=1e-4) else 0
            hls_lambda = hlsutils.hls_lambda(new_phi, self.env.rewards)
            rank_phi = hlsutils.rank(new_phi, tol=1e-4)
            if self.use_tb:
                self.writer.add_scalar('max miss-specification', max_err.cpu().item(), self.t)
                self.writer.add_scalar('hls_lambda', hls_lambda, self.t)
                self.writer.add_scalar('hls_rank', hls_rank, self.t)
                self.writer.add_scalar('hls?', ishls, self.t)
                self.writer.add_scalar('total rank', rank_phi, self.t)
            if self.use_wandb:
                wandb.log({'max miss-specification': max_err.cpu().item(),
                           'hls_lambda': hls_lambda,
                           'hls_rank': hls_rank,
                           'hls?': ishls,
                           'total rank': rank_phi}, step=self.t)


    # def _post_train(self, loader=None):
    #     with torch.no_grad():
    #
    #         dim = self.model.embedding_dim
    #         f, r = self.buffer.get_all()
    #         # inv_A = torch.eye(dim) / self.ucb_regularizer
    #         torch_feat = torch.tensor(f, dtype=TORCH_FLOAT)
    #         torch_rew = torch.tensor(r.reshape(-1,1), dtype=TORCH_FLOAT)
    #         torch_phi = self.model.embedding(torch_feat)
    #         YYtt = (torch_phi.T @ torch_phi + self.ucb_regularizer *torch.eye(dim))
    #         BBtt = torch.sum(torch_phi * torch_rew, 0)
    #         thetatt = torch.linalg.solve(YYtt, BBtt)
    #         self.theta = thetatt
    #         self.A = YYtt
    #         self.b_vec = BBtt
    #         self.inv_A = torch.linalg.inv(self.A)
    #         self.A_logdet = torch.logdet(self.A)
    #         # for el in torch_phi:
    #         #     # el = el.squeeze()
    #         #     Au = inv_A @ el
    #         #     den = 1 + torch.dot(el.T, Au)
    #         #     inv_A -= torch.outer(Au, Au) / (den)
    #         # theta2 = inv_A @ BBtt
    #
    #         # # numpy
    #         # np_f_32 = np.float32(torch_phi.numpy())
    #         # np_r_32 = np.float32(torch_rew.numpy())
    #         # np_inv_A_32 = np.float32(np.eye(dim)) / self.ucb_regularizer
    #         # np_b_32 = np.sum(np_f_32 * np_r_32, 0)
    #         # for el in np_f_32:
    #         #     Au = np_inv_A_32 @ el
    #         #     den = 1 + np.dot(el.T, Au)
    #         #     np_inv_A_32 -= np.outer(Au, Au) / (den)
    #         # np_theta_32 = np_inv_A_32 @ np_b_32
    #
    #
    #
    #         # # A = np.eye(dim) * self.ucb_regularizer
    #         # self.b_vec = torch.zeros(dim).to(self.device)
    #         # self.inv_A = torch.eye(dim).to(self.device) / self.ucb_regularizer
    #         # self.A = torch.zeros_like(self.inv_A)
    #         # self.features_bound = 0
    #         # for b_features, b_rewards, b_weights in loader:
    #         #     phi = self.model.embedding(b_features) #.cpu().detach().numpy()
    #
    #         #     # features
    #         #     max_norm = torch.norm(phi, p=2, dim=1).max().cpu()
    #         #     self.features_bound = max(self.features_bound, max_norm)
    #         #     self.b_vec = self.b_vec + (phi * b_rewards).sum(dim=0)
    #         #     #SM
    #         #     for v in phi:
    #         #         Au = self.inv_A @ v
    #         #         den = 1 + torch.dot(v.T, Au)
    #         #         self.inv_A -= torch.outer(Au, Au) / (den)
    #         #         self.A += torch.outer(v.ravel(),v.ravel())
    #         # #     A = A + np.sum(phi[...,None]*phi[:,None], axis=0)
    #         # # # strange issue with making operations directly in pytorch
    #         # # self.inv_A = torch.tensor(np.linalg.inv(A), dtype=torch.float)
    #         # self.theta = self.inv_A @ self.b_vec
    #         self.features_bound = torch.norm(torch_phi, p=2, dim=1).max().cpu().item()
    #         self.param_bound = torch.linalg.norm(self.theta, 2).item()
    #         self.writer.add_scalar('param_bound', self.param_bound, self.t)
    #         self.writer.add_scalar('features_bound', self.features_bound, self.t)
    #         # min_eig = torch.linalg.eigvalsh(self.A/(self.t+1)).min() / self.features_bound
    #         # self.writer.add_scalar('min_eig_empirical_design', min_eig, self.t)
    #
    #         pred = torch_phi @ self.theta
    #         mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
    #         self.writer.add_scalar('mse_linear', mse_loss.item(), self.t)
    #         # print(f"{self.t}mse1 (used): ", mse_loss.item())
    #
    #         # pred = torch_phi @ thetatt
    #         # mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
    #         # print(f"{self.t}mse2: ", mse_loss.item())
    #
    #         # pred = torch_phi @ theta2
    #         # mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
    #         # print(f"{self.t}mse3: ", mse_loss.item())
    #
    #         # jjj = torch.linalg.inv(YYtt)
    #         # pred = torch_phi @ (jjj @ BBtt)
    #         # mse_loss = F.mse_loss(pred.reshape(-1,1), torch_rew)
    #         # print(f"{self.t}mse4: ", mse_loss.item())
    #
    #         # pred = np_f_32 @ np_theta_32
    #         # mse_loss = ((pred - np_r_32)**2).mean()
    #         # print(f"{self.t}mse5(np 32): ", mse_loss)
    #
    #
    #         # self.inv_A = jjj
    #         # self.b_vec = BBtt
    #         # self.theta = jjj @ BBtt
    #
    #         # debug metric
    #         if hasattr(self.env, 'feature_matrix'):
    #             # compute misspecification error on all samples
    #             nc,na,nd = self.env.feature_matrix.shape
    #             U = self.env.feature_matrix.reshape(-1, nd)
    #             xt = torch.tensor(U, dtype=TORCH_FLOAT)
    #             H = self.model.embedding(xt)
    #             newfeatures = H.reshape(nc, na, self.model.embedding_dim)
    #             newreward = newfeatures @ self.theta
    #             max_err = np.abs(self.env.rewards - newreward.cpu().detach().numpy()).max()
    #             self.writer.add_scalar('max miss-specification', max_err, self.t)
    #
    #             # IS HLS
    #             newfeatures = newfeatures.cpu().detach().numpy()
    #             hls_rank = hlsutils.hls_rank(newfeatures, self.env.rewards)
    #             ishls = 1 if hlsutils.is_hls(newfeatures, self.env.rewards) else 0
    #             hls_lambda = hlsutils.hls_lambda(newfeatures, self.env.rewards)
    #             rank_phi = hlsutils.rank(newfeatures, None)
    #             self.writer.add_scalar('hls_lambda', hls_lambda, self.t)
    #             self.writer.add_scalar('hls_rank', hls_rank, self.t)
    #             self.writer.add_scalar('hls?', ishls, self.t)
    #             self.writer.add_scalar('rank_phi', rank_phi, self.t)




