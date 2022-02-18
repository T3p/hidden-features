import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Any

@dataclass
class LinearContinuous:

    context_dim: int
    num_actions: int
    context_generation: Optional[str] = "uniform"
    feature_expansion: Optional[str] = None
    seed: Optional[int] = 0
    max_value: Optional[int] = 1
    min_value: Optional[int] = -1
    features_sigma: Optional[int] = 1
    features_mean: Optional[Any] = 0
    feature_proba: Optional[float] = 0.5
    min_value: Optional[int] = -1
    noise: Optional[str]=None
    noise_param: Optional[Any]=None

    def __post_init__(self) -> None:
        assert self.context_generation in ["gaussian", "bernoulli", "uniform"]
        assert self.feature_expansion in [None, "expanded", "onehot"]
        self.np_random = np.random.RandomState(seed=self.seed)
        if self.feature_expansion is None:
            self.feature_dim = self.context_dim
        elif self.feature_expansion == "expanded":
            self.feature_dim = self.context_dim * self.num_actions
        elif self.feature_expansion == "onehot":
            self.feature_dim = self.context_dim + self.num_actions
        self.theta = self.np_random.uniform(low=self.min_value, high=self.max_value, size=self.feature_dim)

    def _sample_context(self) -> np.ndarray:
        if self.context_generation == "uniform":
            self.context = self.np_random.uniform(low=self.min_value, high=self.max_value, size=(self.context_dim, ))
        elif self.context_generation == "gaussian":
            self.context = self.np_random.randn(self.context_dim) * self.features_sigma + self.features_mean
        elif self.context_generation == "bernoulli":
            self.context = self.np_random.binomial(1, p=self.feature_proba, size=self.context_dim)
        return self.context

    def features(self) -> np.ndarray:
        """ sample a context and return its expanded feature
        """
        context = self._sample_context().copy()
        if self.feature_expansion is None:
            self.feat = context
        elif self.feature_expansion == "expanded":
            feat = np.zeros((self.num_actions, self.feature_dim))
            for a in range(self.num_actions):
                feat[a, a * self.context_dim: a * self.context_dim + self.context_dim] = context
            self.feat = feat
        elif self.feature_expansion == "onehot":
            self.feat = np.zeros((self.num_actions, self.feature_dim))
            for a in range(self.num_actions):
                feat[a, 0:self.context_dim] = context
                feat[a, a] = 1
        return self.feat

    def step(self, action: int) -> float:
        reward = np.dot(self.theta, self.feat[action])
        if self.noise is not None:
            if self.noise == "bernoulli":
                reward = self.np_random.binomial(n=1, p=reward).item()
            else:
                reward = reward + self.np_random.randn(1).item() * self.noise_param     
        return reward

    def best_reward_and_action(self) -> Tuple[int, float]:
        """ Best action and reward in the current context
        """
        rewards = self.feat @ self.theta
        action = np.argmax(rewards)
        return rewards[action], action
    
    def expected_reward(self, action: int) -> float:
        reward = np.dot(self.theta, self.feat[action])
        return reward

