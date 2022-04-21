import imp
import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .nnlinucb import NNLinUCB


class NNEpsGreedy(NNLinUCB):
    def __init__(
        self, env: Any, model: nn.Module,
            device: Optional[str] = "cpu",
            batch_size: Optional[int] = 256,
            max_updates: Optional[int] = 1,
            learning_rate: Optional[float] = 0.001,
            weight_decay: Optional[float] = 0,
            buffer_capacity: Optional[int] = 10000,
            seed: Optional[int] = 0,
            reset_model_at_train: Optional[bool] = True,
            update_every_n_steps: Optional[int] = 100,
            epsilon_min: float=0.05,
            epsilon_start: float=2,
            epsilon_decay: float=200,
            time_random_exp: int=0,
            ucb_regularizer: Optional[float]=1
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, reset_model_at_train, update_every_n_steps, 0, 0.01, ucb_regularizer, 1, 0)
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.time_random_exp = time_random_exp
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
            xt = torch.tensor(features).to(self.device)
            phi = self.model.embedding(xt)
            scores = phi @ self.theta
            action = torch.argmax(scores).item()
        assert 0 <= action < self.env.action_space.n
        return action
