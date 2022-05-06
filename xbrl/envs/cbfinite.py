import pdb
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import sklearn
from scipy.special import expit as sigmoid
from .spaces import DiscreteFix


@dataclass
class CBFinite:

    feature_matrix: np.ndarray=None # n_contexts x n_actions x dim
    rewards: np.ndarray=None # n_contexts x n_actions
    noise: Optional[str]=None
    noise_param: Optional[Any]=None
    shuffle: Optional[bool]=False
    seed: Optional[int]=None
    dataset_name: Optional[str]="linearfinite"

    def __post_init__(self) -> None:
        self.np_random = np.random.RandomState(seed=self.seed)
        if self.shuffle:
            self.feature_matrix, self.rewards = sklearn.utils.shuffle(self.feature_matrix, self.rewards, random_state=self.seed)
        assert self.noise in [None, "bernoulli", "gaussian"]
        self.idx = -1
        assert len(self.feature_matrix.shape) == 3
        assert (self.feature_matrix.shape[0] == self.rewards.shape[0]) and (self.feature_matrix.shape[1] == self.rewards.shape[1])
        self.feature_dim = self.feature_matrix.shape[-1]
        self.action_space = DiscreteFix(n=self.feature_matrix.shape[1])

        if self.noise == "bernoulli":
            self.rewards = sigmoid(self.rewards)

        self.n_contexts, self.n_arms, self.dim = self.feature_matrix.shape

        # # increase artificially the minimum gap
        # for x in range(self.__len__()):
        #     best_action = np.argmax(self.rewards[x])
        #     self.rewards[x, best_action] += 1


    def sample_context(self) -> np.ndarray:
        self.idx += 1
        if self.idx == self.__len__():
            self.idx = 0  
        return self.feature_matrix[self.idx]

    def features(self) -> np.ndarray:
        """ sample a context and return its expanded feature
        """
        return self.sample_context()
    
    def step(self, action: int) -> float:
        reward = self.rewards[self.idx, action]
        if self.noise is not None:
            if self.noise == "bernoulli":
                reward = self.np_random.binomial(n=1, p=reward)
            else:
                reward = reward + self.np_random.randn(1).item() * self.noise_param     
        return reward

    def __len__(self) -> int:
        return self.feature_matrix.shape[0]
    
    def best_reward_and_action(self) -> Tuple[int, float]:
        best_action = np.argmax(self.rewards[self.idx])
        best_reward = self.rewards[self.idx, best_action]
        return best_reward, best_action

    def expected_reward(self, action: int) -> float:
        return self.rewards[self.idx, action]

    def min_suboptimality_gap(self):
        min_gap = np.inf
        for ctx in range(self.__len__()):
            arr = sorted(self.rewards[ctx])
            for i in range(self.n_arms-1):
                diff = arr[i+1] - arr[i]
                if diff <= min_gap and diff > 0:
                    min_gap = diff
        return min_gap

