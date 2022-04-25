import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from ... import TORCH_FLOAT
from .nnlinucbinc import NNLinUCBInc

class NNEGInc(NNLinUCBInc):
    
    def __init__(
        self,
        env: Any,
        model: nn.Module,
        device: Optional[str]="cpu",
        batch_size: Optional[int]=256,
        max_updates: Optional[int]=1,
        learning_rate: Optional[float]=0.001,
        weight_decay: Optional[float]=0,
        buffer_capacity: Optional[int]=10000,
        seed: Optional[int]=0,
        update_every: Optional[int] = 100,
        ucb_regularizer: Optional[float]=1,
        epsilon_min: float=0.05,
        epsilon_start: float=2,
        epsilon_decay: float=200,
        time_random_exp: int=0,
        use_tb: Optional[bool]=True,
        use_wandb: Optional[bool]=False
    ) -> None:
        noise_std, delta, bonus_scale = 0, 0.001, 0
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, update_every, noise_std, delta, ucb_regularizer, bonus_scale, use_tb, use_wandb)
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
            xt = torch.tensor(features, dtype=TORCH_FLOAT).to(self.device)
            phi = self.model.embedding(xt)
            scores = phi @ self.theta
            action = torch.argmax(scores).item()
        assert 0 <= action < self.env.action_space.n
        return action
