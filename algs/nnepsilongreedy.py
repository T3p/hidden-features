import imp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F

from .xbdiscrete import XBTorchDiscrete

@dataclass
class NNEpsGreedy(XBTorchDiscrete):

    epsilon_min: float=0.05
    epsilon_start: float=2
    epsilon_decay: float=200

    def __post_init__(self) -> None:
        self.epsilon = self.epsilon_start
        self.np_random = np.random.RandomState(self.seed)

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
