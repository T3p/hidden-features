import numpy as np
from typing import Optional, Any, Callable
from .batched.templates import XBModule
from .linear import inv_sherman_morrison
from scipy.optimize import root

def sigmoid(x):
    return np.where(x < 0, np.exp(x)/(np.exp(x) + 1), 1/(np.exp(-x) + 1))

def sigmoid_fod(x):
    return sigmoid(x) * (-sigmoid(x) + 1)

def sigmoid_sod(x):
    return sigmoid_fod(x) * (-2 * sigmoid(x) + 1)

def sigmoid_nonlinearity(param_bound, features_bound):
    z = (1 + param_bound) * features_bound
    return 2 * (np.cosh(z) + 1)

class UCBGLM(XBModule):

    def __init__(
        self, env: Any,
        seed: Optional[int] = 0, 
        update_every_n_steps: Optional[int] = 1,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.,
        link_function: Callable[[float], float] = sigmoid,
        nonlinearity_function: Callable[[float, float], float] = sigmoid_nonlinearity,
        link_fod: Optional[Callable[[float], float]] = sigmoid_fod,
        link_sod: Optional[Callable[[float], float]] = sigmoid_sod
    ) -> None:
        super().__init__(env, None, None, None, None, None, None, 0, seed, None, update_every_n_steps)
        self.np_random = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale
        self.link_function = link_function
        self.nonlinearity_function = nonlinearity_function
        self.link_fod = link_fod
        self.link_sod_sod = link_sod
        
    def reset(self) -> None:
        super().reset()
        dim = self.env.feature_dim
        self.b_vec = np.zeros(dim)
        self.inv_A = np.eye(dim) / self.ucb_regularizer
        self.theta = np.zeros(dim)
        self.new_b_vec = np.zeros(dim)
        self.new_inv_A = np.eye(dim) / self.ucb_regularizer
        #self.new_theta = np.zeros(dim)
        self.param_bound = 1
        self.features_bound = 1
        self.features_history = []

    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.env.action_space.n
        dim = features.shape[1]
        nonlinearity_coeff = self.nonlinearity_function(self.param_bound, self.features_bound)
        beta = nonlinearity_coeff * self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2*self.t/self.ucb_regularizer)/self.delta)) + self.param_bound * np.sqrt(self.ucb_regularizer)

        # get features for each action and make it tensor
        bonus = ((features @ self.inv_A)*features).sum(axis=1)
        bonus = self.bonus_scale * beta * np.sqrt(bonus)
        ucb = features @ self.theta + bonus
        action = np.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        assert 0 <= action < self.env.action_space.n, ucb

        return action

    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        # exp = (features, reward)
        # self.buffer.append(exp)

        # estimate linear component on the embedding + UCB part
        v = features
        self.features_history.append(features)
        self.features_bound = max(self.features_bound, np.linalg.norm(v, 2).item())
        self.writer.add_scalar('features_bound', self.features_bound, self.t)

        self.new_b_vec = self.new_b_vec + v * reward
        self.new_inv_A, den = inv_sherman_morrison(v, self.new_inv_A)
        # self.A_logdet += np.log(den)
        
        #self.new_theta = self.inv_A @ self.b_vec

    def train(self) -> float:
        if self.t % self.update_every_n_steps == 0:
            self.inv_A = self.new_inv_A
            #self.theta = self.new_theta
            self.b_vec = self.new_b_vec
            
            def mle_equation(theta):
                feature_matrix = np.stack(self.features_history, axis=0) #txd
                predictions = self.link_function(feature_matrix @ theta) #t
                value = predictions @ feature_matrix - self.b_vec # d
                derivatives = self.link_fod(feature_matrix @ theta) #t
                jac = (derivatives * feature_matrix.T) @ feature_matrix #dxd
                return value, jac
            
            sol = root(mle_equation, self.theta, jac=True)
            if sol.success:
                self.theta = sol.x
                self.param_bound = np.linalg.norm(self.theta, 2).item()
                self.writer.add_scalar('param_bound', self.param_bound, self.t)
        
        return 0

