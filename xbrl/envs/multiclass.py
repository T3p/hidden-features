from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from .spaces import DiscreteFix
from sklearn.utils import shuffle

@dataclass
class MulticlassToBandit:
    
    X: np.ndarray
    y: np.ndarray
    dataset_name: Optional[str] = None
    seed: Optional[int] = 0
    noise: Optional[str] = None
    noise_param: Optional[float] = None
    shuffle: Optional[bool] = True

    def __post_init__(self) -> None:
        """Initialize Class"""
        # self.y = (rankdata(self.y, "dense") - 1).astype(int)
        self.y = OrdinalEncoder(dtype=int).fit_transform(self.y.reshape((-1, 1)))
        self.action_space = DiscreteFix(n=np.unique(self.y).shape[0])
        self.np_random = np.random.RandomState(seed=self.seed)
        if shuffle:
            self.X, self.y = shuffle(self.X, self.y, random_state=self.seed)
        assert self.noise in [None, "bernoulli", "gaussian"]
        self.idx = -1

    def sample_context(self) -> np.ndarray:
        # self.idx = self.np_random.randint(0, self.__len__(), 1).item()
        self.idx += 1
        if self.idx == self.__len__():
            self.idx = 0  
        return self.X[self.idx]

    def step(self, action: int) -> float:
        """ Return a realization of the reward in the context for the selected action
        """
        assert self.action_space.contains(action), action
        reward = 1. if self.y[self.idx] == action else 0.
        if self.noise is not None:
            if self.noise == "bernoulli":
                proba = reward + self.noise_param if reward == 0 else reward - self.noise_param
                reward = self.np_random.binomial(n=1, p=proba).item()
            else:
                reward = reward + self.np_random.randn(1).item() * self.noise_param        
        return reward
    
    def best_reward_and_action(self) -> Tuple[int, float]:
        """ Best action and reward in the current context
        """
        action = self.y[self.idx]
        return 1., action
    
    def expected_reward(self, action: int) -> float:
        assert self.action_space.contains(action)
        return 1. if self.y[self.idx] == action else 0.

    def __len__(self) -> int:
        return self.y.shape[0]
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if (self.idx >= self.__len__()):
            raise StopIteration
        self.idx += 1
        return self.__getitem__(self.idx)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

@dataclass
class MCOneHot(MulticlassToBandit):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.eye = np.eye(self.action_space.n)
        self.feature_dim = self.X.shape[1] + self.action_space.n

    def __getitem__(self, idx):
        context = self.X[idx]
        na = self.action_space.n  
        tile_p = [na] + [1]*len(context.shape)
        x = np.tile(context, tile_p)
        x_y = np.hstack((x, self.eye))
        rwd = np.zeros((self.action_space.n,))
        rwd[self.y[idx]] = 1
        return x_y, rwd

    def features(self) -> np.ndarray:
        """Return a feature representation obtained from the concatenation
            of the current context with a one-hot-encoding representation of the features
        """
        return self.__getitem__(self.idx)[0]

@dataclass
class MCExpanded(MulticlassToBandit):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_dim = self.X.shape[1] * self.action_space.n
    
    def __getitem__(self, idx):
        context = self.X[idx]
        na = self.action_space.n
        act_dim = self.X.shape[1]
        F = np.zeros((na, self.feature_dim))
        for a in range(na):
            F[a, a * act_dim:a * act_dim + act_dim] = context
        rwd = np.zeros((self.action_space.n,))
        rwd[self.y[idx]] = 1
        return F, rwd

    def features(self) -> np.ndarray:
        """Return a feature representation obtained from the concatenation
            of the current context with a one-hot-encoding representation of the features
        """
        return self.__getitem__(self.idx)[0]
