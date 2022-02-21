import imp
import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .templates import XBModule

class NNEpsGreedy(XBModule):

    def __init__(
        self, env: Any, model: nn.Module, device: Optional[str] = "cpu", batch_size: Optional[int] = 256, max_updates: Optional[int] = 1, learning_rate: Optional[float] = 0.001, weight_decay: Optional[float] = 0, buffer_capacity: Optional[int] = 10000, seed: Optional[int] = 0, reset_model_at_train: Optional[bool] = True, update_every_n_steps: Optional[int] = 100,
        epsilon_min: float=0.05,
        epsilon_start: float=2,
        epsilon_decay: float=200,
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, reset_model_at_train, update_every_n_steps)
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.np_random = np.random.RandomState(self.seed)

    def reset(self) -> None:
        super().reset()
        self.epsilon = self.epsilon_start

    def _train_loss(self, b_features, b_rewards):
        # MSE LOSS
        prediction = self.model(b_features)
        mse_loss = F.mse_loss(prediction, b_rewards)
        self.writer.add_scalar('mse_loss', mse_loss, self.batch_counter)
        return mse_loss
    
    @torch.no_grad()
    def play_action(self, features: np.ndarray) -> int:
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_start - self.epsilon_min) / self.epsilon_decay
        self.writer.add_scalar('epsilon', self.epsilon, self.t)
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.choice(self.env.action_space.n, size=1).item()
        else:
            xt = torch.FloatTensor(features).to(self.device)
            scores = self.model(xt)
            action = torch.argmax(scores).item()
        return action
