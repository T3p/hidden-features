from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from sklearn.utils import shuffle


@dataclass
class CBFinite:

    features: np.ndarray # n_contexts x n_actions x dim
    rewards: np.ndarray # n_contexts x n_actions
    noise: Optional[str]=None
    noise_param: Optional[Any]=None
    shuffle: Optional[bool]=False

    def __post_init__(self) -> None:
        self.np_random = np.random.RandomState(seed=self.seed)
        if shuffle:
            self.features, self.rewards = shuffle(self.features, self.rewards, random_state=self.seed)
        assert self.noise in [None, "bernoulli", "gaussian"]
        self.idx = -1
        assert len(self.features.shape) == 3

    def _sample_context(self) -> np.ndarray:
        self.idx += 1
        if self.idx == self.__len__():
            self.idx = 0  
        return self.features[self.idx]

    def features(self) -> np.ndarray:
        """ sample a context and return its expanded feature
        """
        return self.features
    
    def step(self, action: int) -> float:
        reward = self.rewards[self.idx, action]
        if self.noise is not None:
            if self.noise == "bernoulli":
                reward = self.np_random.binomial(n=1, p=reward).item()
            else:
                reward = reward + self.np_random.randn(1).item() * self.noise_param     
        return reward

    def __len__(self) -> int:
        return self.features.shape[0]
    
    def best_reward_and_action(self) -> Tuple[int, float]:
        best_action = np.argmax(self.rewards[self.idx])
        best_reward = self.rewards[self.idx,best_action]
        return best_reward, best_action

    def expected_reward(self, action: int) -> float:
        return self.rewards[self.idx, action]
