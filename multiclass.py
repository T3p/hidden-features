from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np

from sklearn.utils import check_X_y
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

@dataclass
class DiscreteFix:
    n: int

    def contains(self, x) -> bool:
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)  # type: ignore
        else:
            return False
        return 0 <= as_int < self.n

@dataclass
class MulticlassToBandit:
    
    X: np.ndarray
    y: np.ndarray
    dataset_name: Optional[str] = None
    seed: Optional[int] = 0
    noise: Optional[str] = None
    noise_param: Optional[float] = None

    def __post_init__(self):
        """Initialize Class"""
        self.X, y = check_X_y(X=self.X, y=self.y, ensure_2d=True, multi_output=False)
        # re-index actions from 0 to n_classes
        self.y = (rankdata(y, "dense") - 1).astype(int)
        self.action_space = DiscreteFix(n=np.unique(self.y).shape[0])
        self.np_random = np.random.RandomState(seed=self.seed)
        assert self.noise in [None, "bernoulli", "gaussian"]

    def split_pretrain_test(
        self,
        test_size: Union[int, float] = 0.75,
        random_state: Optional[int] = None,
    ) -> Tuple[MulticlassToBandit]:
        """Split the original data into the pretraining (used for hyperparameter optimization) and test (used for online test) sets.
        Parameters
        ----------
        test_size: float or int, default=0.75
            If float, should be between 0.0 and 1.0 and represent the proportion of the data to include in the test split.
            If int, represents the absolute number of test samples.
        random_state: int, default=None
            Controls the random seed in pretrain-test split.

        Returns
        -------
        xb_pretraining: MulticlassToBandit
            Instance constructed using pretraining samples
        xb_test: MulticlassToBandit
            Instance constructed using test samples
        """
        (
            X_pre,
            X_test,
            y_pre,
            y_test
        ) = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        return MulticlassToBandit(
                X=X_pre, y=y_pre, 
                dataset_name=f'{self.dataset_name}_pretraining' if self.dataset_name else None,
                noise=self.noise, noise_param=self.noise_param, seed=self.seed
            ), MulticlassToBandit(
                X=X_test, y=y_test, 
                dataset_name=f'{self.dataset_name}_test' if self.dataset_name else None,
                noise=self.noise, noise_param=self.noise_param, seed=self.seed
            ), 

    def sample_context(self) -> np.ndarray:
        self.idx = self.np_random.randint(0, self.n_samples, 1).item()
        return self.X[self.idx]

    def step(self, action: int) -> float:
        """ Return a realization of the reward in the context for the selected action
        """
        assert self.action_space.contains(action)
        reward = self.y[self.idx] != action
        if self.noise is not None:
            if self.noise == "bernoulli":
                proba = reward + self.noise_param if reward == 0 else reward - self.noise_param
                reward = self.np_random.binomial(n=1, p=proba)
            else:
                reward = reward + self.np_random.randn(1) * self.noise_param        
        return reward
    
    def best_reward(self) -> float:
        """ Maximum reward in the current context
        """
        return 1
    
    def expected_reward(self, action: int) -> float:
        assert self.action_space.contains(action)
        return self.y[self.idx] != action

    @property
    def n_samples(self) -> int:
        return self.y.shape[0]
        