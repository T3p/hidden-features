import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class LinearBandit:

    context_dim: int
    noise: float
    num_actions: int

    def __post_init__(self) -> None:
        self.theta = np.random.uniform(low=-1, high=1, size=(self.num_actions, self.context_dim))

    def _sample_context(self) -> np.ndarray:
        self.context = np.random.uniform(low=-1, high=1, size=(self.context_dim, ))
        return self.context

    def sample(self) -> np.ndarray:
        """ sample a context and return its expanded feature
        """
        feature_dim = self.context_dim * self.num_actions
        context = self._sample_context().copy()
        feat = np.zeros((self.num_actions, feature_dim))
        for a in range(self.num_actions):
            feat[a, a * self.context_dim: a * self.context_dim + self.context_dim] = context
        self.feat = feat
        return feat

    def step(self, action: int) -> float:
        reward = np.dot(self.theta[action], self.context)
        noise = self.noise * np.random.randn()
        return reward + noise

    def best_reward_and_action(self) -> Tuple[int, float]:
        """ Best action and reward in the current context
        """
        rewards = np.dot(self.theta, self.context)
        action = np.argmax(rewards)
        return rewards[action], action
    
    def expected_reward(self, action: int) -> float:
        rewards = np.dot(self.theta, self.context)
        return rewards[action]
        