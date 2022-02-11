import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .spaces import DiscreteFix

@dataclass
class Bandit_Linear:

    feature_dim: int
    noise: float
    arms: int
    seed: Optional[int]=0

    def __post_init__(self) -> None:
        self.theta = np.random.uniform(low=-1, high=1, size=(self.feature_dim,))
        self.np_random = np.random.RandomState(self.seed)
        self.action_space = DiscreteFix(n=self.arms)

    def sample_context(self) -> np.ndarray:
        self.phi = self.np_random.uniform(low=-1, high=1, size=(self.arms, self.feature_dim))
        return self.phi

    def features(self) -> np.ndarray:
        return self.phi

    def step(self, action: int) -> float:
        reward = self.phi[action].dot(self.theta)
        noise = self.np_random.randn(1).item() * self.noise
        return reward + noise

    def best_reward_and_action(self) -> Tuple[int, float]:
        """ Best action and reward in the current context
        """
        rewards = self.phi @ self.theta
        action = np.argmax(rewards)
        return rewards[action], action
    
    def expected_reward(self, action: int) -> float:
        rewards = self.phi @ self.theta
        return rewards[action]
        