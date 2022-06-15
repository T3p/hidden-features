import pdb

import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from omegaconf import DictConfig
import wandb

from .nnlinucb import NNLinUCB, TORCH_FLOAT


class GradientUCB(NNLinUCB):
    def __init__(
            self, env: Any,
            cfg: DictConfig,
            model: Optional[nn.Module] = None
    ) -> None:
        super().__init__(env, cfg, model)
        self.time_random_exp = cfg.time_random_exp
        self.np_random = np.random.RandomState(self.seed)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.Z = self.ucb_regularizer * torch.ones((self.num_params, ), dtype=TORCH_FLOAT, device=self.device)

    def _post_train(self, loader=None) -> None:
        pass

    def play_action(self, features: np.ndarray) -> int:
        self.is_random_step = 1 # actually it means no glrt action
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
        predicted_rewards = self.model(features_tensor).squeeze()
        gs = []
        ucbs = []
        bonuses = []
        for reward in predicted_rewards:
            self.model.zero_grad()
            reward.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.model.parameters()])
            gs.append(g)
            bonus = g * g / self.Z
            bonus = self.bonus_scale * torch.sqrt(torch.sum(bonus))
            ucb = (reward + bonus).item()
            bonuses.append(bonus.item())
            ucbs.append(ucb)
        action = np.argmax(ucbs)
        self.Z += gs[action] * gs[action]

        if self.use_tb:
            self.writer.add_scalar('bonus selected action', bonuses[action], self.t)
        if self.use_wandb:
            wandb.log({'bonus selected action': bonuses[action]}, step=self.t)

        return action



