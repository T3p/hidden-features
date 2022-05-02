import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from ... import TORCH_FLOAT
from .nnlinucbinc import NNLinUCBInc
from omegaconf import DictConfig

class NNEGInc(NNLinUCBInc):
    
    def __init__(
        self,
        env: Any,
        model: nn.Module,
        cfg: DictConfig
    ) -> None:
        noise_std, delta, bonus_scale = 0, 0.001, 0
        super().__init__(env, model, cfg)
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_decay = cfg.epsilon_decay
        self.time_random_exp = cfg.time_random_exp
        self.np_random = np.random.RandomState(self.seed)

    def reset(self) -> None:
        super().reset()
        self.epsilon = self.epsilon_start

    @torch.no_grad()
    def play_action(self, features: np.ndarray) -> int:
        if self.t > self.time_random_exp and self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_start - self.epsilon_min) / self.epsilon_decay
        self.writer.add_scalar('epsilon', self.epsilon, self.t)
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.choice(self.env.action_space.n, size=1).item()
        else:
            xt = torch.tensor(features, dtype=TORCH_FLOAT).to(self.device)
            phi = self.model.embedding(xt)
            scores = phi @ self.theta
            action = torch.argmax(scores).item()
        assert 0 <= action < self.env.action_space.n
        return action
