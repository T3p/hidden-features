import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from .spaces import DiscreteFix
from scipy.special import expit as sigmoid

def normalize_linrep(features, param, scale=1.):
    param_norm = np.linalg.norm(param)
    new_param = param / param_norm * scale
    new_features = features * param_norm / scale
    return new_features, new_param

def random_transform(features, param, normalize=True, seed=0):
    rng = np.random.RandomState(seed)
    dim = len(param)
    A = rng.normal(size=(dim, dim))

    A = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(A)
    
    new_features = features @ q
    new_param = q.T @ param
        
    if normalize:
        new_features, new_param = normalize_linrep(new_features, new_param)
    
    val = features @ param - new_features @ new_param
    assert np.allclose(features @ param, new_features @ new_param)
    return new_features, new_param

def derank_hls(features, param, newrank=1, transform=True, normalize=True, seed=0):
    nc = features.shape[0]

    rewards = features @ param
    # compute optimal arms
    opt_arms = np.argmax(rewards, axis=1)
    # compute features of optimal arms
    opt_feats = features[np.arange(nc), opt_arms, :]
    opt_rews = rewards[np.arange(nc), opt_arms].reshape((nc, 1)) 
    remove = min(max(nc - newrank + 1, 0), nc)
    
    new_features = np.array(features)
    outer = np.matmul(opt_rews[:remove], opt_rews[:remove].T)
    xx = np.matmul(outer, opt_feats[:remove, :]) \
        / np.linalg.norm(opt_rews[:remove])**2
    new_features[np.arange(remove), opt_arms[:remove], :] = xx
    
    new_param = param.copy()
    
    if transform:
        new_features, new_param = random_transform(new_features, new_param, normalize=normalize, seed=seed)
    elif normalize:
        new_features, new_param = normalize_linrep(new_features, new_param, seed=seed)
        
    assert np.allclose(features @ param, new_features @ new_param)
    return new_features, new_param

@dataclass
class LinearContinuous:

    context_dim: int
    num_actions: int
    context_generation: Optional[str] = "uniform"
    feature_expansion: Optional[str] = None
    seed: Optional[int] = 0
    seed_problem: Optional[int]=99
    max_value: Optional[int] = 1
    min_value: Optional[int] = -1
    features_sigma: Optional[int] = 1
    features_mean: Optional[Any] = 0
    feature_proba: Optional[float] = 0.5
    min_value: Optional[int] = -1
    noise: Optional[str]=None
    noise_param: Optional[Any]=None
    dataset_name: Optional[str]="linearcontinuous"

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
        random_problem = np.random.RandomState(seed=self.seed_problem)
        self.theta = random_problem.uniform(low=self.min_value, high=self.max_value, size=self.feature_dim)
        self.action_space = DiscreteFix(n=self.num_actions)

    def sample_context(self) -> np.ndarray:
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
        if self.feature_expansion is None:
            # reward(a) = \phi(a) * \theta
            # \phi(a) \sim D
            self.feat = np.zeros((self.num_actions, self.context_dim))
            for a in range(self.num_actions):
                self.feat[a] = self.sample_context()
        elif self.feature_expansion == "expanded":
            self.feat = np.zeros((self.num_actions, self.feature_dim))
            for a in range(self.num_actions):
                self.feat[a, a * self.context_dim: a * self.context_dim + self.context_dim] = self.context
        elif self.feature_expansion == "onehot":
            self.feat = np.zeros((self.num_actions, self.feature_dim))
            for a in range(self.num_actions):
                self.feat[a, 0:self.context_dim] = self.context
                self.feat[a, a] = 1
        return self.feat

    def _expected_reward(self, features) -> np.ndarray:
        z = features @ self.theta
        if self.noise == 'bernoulli':
            rewards = sigmoid(z)
        else:
            rewards = z
        return rewards

    def step(self, action: int) -> float:
        reward = self._expected_reward(self.feat)[action]
        if self.noise is not None:
            if self.noise == "bernoulli":
                reward = self.np_random.binomial(n=1, p=reward)
            else:
                reward = reward + self.np_random.randn(1).item() * self.noise_param     
        return reward

    def best_reward_and_action(self) -> Tuple[int, float]:
        """ Best action and reward in the current context
        """
        rewards = self._expected_reward(self.feat)
        action = np.argmax(rewards).item()
        return rewards[action], action
    
    def expected_reward(self, action: int) -> float:
        reward = self._expected_reward(self.feat)[action]
        return reward

    def min_suboptimality_gap(self, n_samples=1000):
        gap = np.inf
        for _ in range(n_samples):
            self.sample_context()
            features = self.features()
            rewards = self._expected_reward(features)
            sorted = np.sort(rewards)
            action_gap = sorted[-1]-sorted[-2]
            gap = min(gap, action_gap)
        return gap
