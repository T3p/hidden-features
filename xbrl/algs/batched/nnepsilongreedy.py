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
            model: nn.Module
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
        self.epsilon = 1 / (self.t + 1)**(1/3)
        if self.use_tb:
            self.writer.add_scalar('epsilon', self.epsilon, self.t)
        if self.use_wandb:
            wandb.log({'epsilon': self.epsilon})
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.choice(self.env.action_space.n, size=1).item()
        else:
            features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
            if self.model is not None:
                with torch.no_grad():
                    phi = self.model.embedding(features_tensor)
            else:
                phi = features_tensor

            scores = torch.matmul(phi, self.theta)
            action = torch.argmax(scores).item()
        assert 0 <= action < self.env.action_space.n
        return action
