import pdb

import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from omegaconf import DictConfig
import wandb

from .nnlinucb import NNLinUCB, TORCH_FLOAT


class SquareCB(NNLinUCB):
    def __init__(
            self, env: Any,
            cfg: DictConfig,
            model: Optional[nn.Module] = None
    ) -> None:
        super().__init__(env, cfg, model)
        self.gamma_exponent = cfg.gamma_exponent
        self.gamma_scale = cfg.gamma_scale
        self.exploration_param = self.env.action_space.n
        self.time_random_exp = cfg.time_random_exp
        self.np_random = np.random.RandomState(self.seed)

    def _post_train(self, loader=None) -> None:
        pass

    def play_action(self, features: np.ndarray) -> int:
        if self.gamma_exponent == "cbrt":
            gamma = self.gamma_scale * np.cbrt(self.t )
        elif self.gamma_exponent == "sqrt":
            gamma = self.gamma_scale * np.sqrt(self.t)
        self.is_random_step = 1 # actually it means no glrt action
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
        with torch.no_grad():
            predicted_rewards = self.model(features_tensor).squeeze()
        predicted_rewards = predicted_rewards.cpu().numpy()
        gap = predicted_rewards.max() - predicted_rewards
        opt_arms = np.where(gap <= self.mingap_clip)[0]
        subopt_arms = np.where(gap > self.mingap_clip)[0]
        prob = np.zeros(self.env.action_space.n)
        prob[subopt_arms] = 1. / (self.exploration_param + gamma * gap[subopt_arms])
        prob[opt_arms] = (1 - prob[subopt_arms].sum()) / len(opt_arms)
        if self.use_tb:
            self.writer.add_scalar('prob_optarms', 1 - prob[subopt_arms].sum(), self.t)
            self.writer.add_scalar('gamma', gamma, self.t)
        if self.use_wandb:
            wandb.log({'prob_optarms':  1 - prob[subopt_arms].sum(),
                       'gamma': gamma}, step=self.t)

        assert np.isclose(prob.sum(), 1)
        action = self.np_random.choice(np.arange(self.env.action_space.n), 1, p=prob).item()
        return action



