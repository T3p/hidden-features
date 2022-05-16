from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from .spaces import DiscreteFix
import sklearn
import pdb

@dataclass
class MulticlassToBandit:
    
    X: np.ndarray
    y: np.ndarray
    rew_optimal: Optional[float] = 1
    rew_suboptimal: Optional[float] = 0
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
        if self.shuffle:
            self.X, self.y = sklearn.utils.shuffle(self.X, self.y, random_state=self.seed)
        assert self.noise in [None, "bernoulli", "gaussian", "none", "None"]
        self.idx = -1

    def sample_context(self) -> np.ndarray:
        # self.idx = self.np_random.randint(0, self.__len__(), 1).item()
        # self.idx += 1
        # if self.idx == self.__len__():
        #     self.idx = 0  
        self.idx = self.np_random.choice(self.__len__(), 1).item()
        return self.X[self.idx]

    def step(self, action: int) -> float:
        """ Return a realization of the reward in the context for the selected action
        """
        assert self.action_space.contains(action), action
        reward = self.rew_optimal if self.y[self.idx] == action else self.rew_suboptimal
        if self.noise not in [None, "none", "None"]:
            if self.noise == "bernoulli":
                reward = self.np_random.binomial(n=1, p=reward, size=1).item()
            else:
                reward = reward + self.np_random.randn(1).item() * self.noise_param        
        return reward
    
    def best_reward_and_action(self) -> Tuple[int, float]:
        """ Best action and reward in the current context
        """
        action = self.y[self.idx]
        return self.rew_optimal, action
    
    def expected_reward(self, action: int) -> float:
        assert self.action_space.contains(action)
        return self.rew_optimal if self.y[self.idx] == action else self.rew_suboptimal

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

    def description(self) -> str:
        desc = f"{self.dataset_name}\n"
        desc += f"n_contexts: {self.__len__()}\n"
        desc += f"context_dim: {self.X.shape[1]}\n"
        desc += f"n_actions: {self.action_space.n}\n"
        desc += f"rewards[subopt, optimal]: [{self.rew_suboptimal}, {self.rew_optimal}]\n"
        desc += f"noise type: {self.noise} (noise param: {self.noise_param})\n"
        desc += f"seed: {self.seed}\n"
        return desc

@dataclass
class MCOneHot(MulticlassToBandit):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.eye = np.eye(self.action_space.n)
        self.feature_dim = self.X.shape[1] + self.action_space.n
        #construct feature matrix and rewards
        A = self.X.reshape((self.X.shape[0], 1, self.X.shape[1]))
        A = np.tile(A, reps=(1, self.action_space.n, 1))
        B = np.eye(self.action_space.n)
        B = B.reshape((1, self.action_space.n, self.action_space.n))
        B = np.tile(B, reps=(self.X.shape[0], 1, 1))
        self.feature_matrix = np.concatenate((A, B), axis=-1)
        assert self.feature_matrix.shape == (self.X.shape[0], self.action_space.n, self.feature_dim)
        self.rewards = np.array([[self.rew_optimal if self.y[i]==j else self.rew_suboptimal for j in range(self.action_space.n)]
                            for i in range(self.X.shape[0])])
        assert self.rewards.shape == (self.X.shape[0], self.action_space.n)
        

    def __getitem__(self, idx):
        context = self.X[idx]
        na = self.action_space.n  
        tile_p = [na] + [1]*len(context.shape)
        x = np.tile(context, tile_p)
        x_y = np.hstack((x, self.eye))
        rwd = np.ones((self.action_space.n,)) * self.rew_suboptimal
        rwd[self.y[idx]] = self.rew_optimal
        return x_y, rwd

    def features(self) -> np.ndarray:
        """Return a feature representation obtained from the concatenation
            of the current context with a one-hot-encoding representation of the features
        """
        return self.__getitem__(self.idx)[0]
        
    def description(self) -> str:
        desc = super().description()
        desc += f"type: onehot\n"
        desc += f"feat dim: {self.feature_dim}"
        return desc

@dataclass
class MCExpanded(MulticlassToBandit):

    def __post_init__(self) -> None:
        super().__post_init__()
        self.feature_dim = self.X.shape[1] * self.action_space.n
        # construct feature matrix and rewards
        na = self.action_space.n
        feature_matrix = np.zeros((self.X.shape[0], na, self.feature_dim))
        for a in range(na):
            feature_matrix[:, a, a * self.X.shape[1]:(a + 1) * self.X.shape[1]] = self.X
        rewards = self.rew_suboptimal * np.ones((self.X.shape[0], na))
        rewards[range(rewards.shape[0]), self.y.ravel()] = self.rew_optimal
        self.feature_matrix = feature_matrix
        self.rewards = rewards
    
    def __getitem__(self, idx):
        context = self.X[idx]
        na = self.action_space.n
        act_dim = self.X.shape[1]
        F = np.zeros((na, self.feature_dim))
        for a in range(na):
            F[a, a * act_dim:a * act_dim + act_dim] = context
        rwd = np.ones((self.action_space.n,)) * self.rew_suboptimal
        rwd[self.y[idx]] = self.rew_optimal
        return F, rwd

    def features(self) -> np.ndarray:
        """Return a feature representation obtained from the concatenation
            of the current context with a one-hot-encoding representation of the features
        """
        return self.__getitem__(self.idx)[0]

    def description(self) -> str:
        desc = super().description()
        desc += f"type: expanded\n"
        desc += f"feat dim: {self.feature_dim}"
        return desc