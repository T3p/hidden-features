import pdb

import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from omegaconf import DictConfig
import wandb

from .nnlinucb import NNLinUCB, TORCH_FLOAT


class NNEpsGreedy(NNLinUCB):
    def __init__(
        self, env: Any,
            cfg: DictConfig,
            model: Optional[nn.Module] = None
    ) -> None:
        super().__init__(env, cfg, model)
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_decay = cfg.epsilon_decay
        self.time_random_exp = cfg.time_random_exp
        self.np_random = np.random.RandomState(self.seed)

        # initialization
        self.epsilon = self.epsilon_start


    def play_action(self, features: np.ndarray) -> int:
        # if self.t > self.time_random_exp and self.epsilon > self.epsilon_min:
            # self.epsilon -= (self.epsilon_start - self.epsilon_min) / self.epsilon_decay
        if self.epsilon_decay == "cbrt":
            self.epsilon = 1. / np.cbrt(self.t + 1)
        elif self.epsilon_decay == "sqrt":
            self.epsilon = 1. / np.sqrt(self.t + 1)
        else:
            raise NotImplementedError()
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
        if self.model is not None:
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
                dim = self.model.embedding_dim
        else:
            phi = features_tensor
            dim = self.env.feature_dim
        predicted_rewards = torch.matmul(phi, self.theta)
        opt_arms = torch.where(predicted_rewards > predicted_rewards.max() - self.mingap_clip)[0]
        subopt_arms = torch.where(predicted_rewards <= predicted_rewards.max() - self.mingap_clip)[0]
        action = opt_arms[torch.randint(len(opt_arms), (1, ))].cpu().item()
        # Generalized Likelihood Ratio Test
        val = - 2 * np.log(self.delta) + dim * np.log(
            1 + 2 * self.t * self.features_bound / (self.ucb_regularizer * dim))
        # val = self.A_logdet - dim * np.log(self.ucb_regularizer) - 2 * np.log(self.delta)
        beta = self.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.ucb_regularizer)

        if len(subopt_arms) == 0:
            min_ratio = beta ** 2 + 1
        else:
            prediction_diff = predicted_rewards[subopt_arms] - predicted_rewards[action]
            phi_diff = phi[subopt_arms] - phi[action]
            weighted_norm = (torch.matmul(phi_diff, self.inv_A) * phi_diff).sum(axis=1)
            likelihood_ratio = (prediction_diff) ** 2 / (2 * weighted_norm.clamp_min(1e-10))
            min_ratio = likelihood_ratio.min()

        if self.use_tb:
            self.writer.add_scalar('epsilon', self.epsilon, self.t)
            self.writer.add_scalar('GRLT', int(min_ratio > self.glrt_scale * beta**2), self.t)
        if self.use_wandb:
            wandb.log({'epsilon': self.epsilon})
            wandb.log({'GRLT': int(min_ratio > self.glrt_scale * beta**2)})
        if min_ratio > self.glrt_scale * beta**2 or self.np_random.rand() > self.epsilon:
            return action
        else:
            return self.np_random.choice(self.env.action_space.n, size=1).item()

