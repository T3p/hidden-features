import numpy as np
from typing import Optional, Any
from .batched.templates import XBModule

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + np.dot(u.T, Au)
    A_inv -= np.outer(Au, Au) / (den)
    return A_inv, den

class LinUCB(XBModule):

    def __init__(
        self,
        env: Any,
        seed: Optional[int] = 0, 
        update_every_n_steps: Optional[int] = 1,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.
    ) -> None:
        super().__init__(env, None, None, None, None, None, None, 0, seed, None, update_every_n_steps)
        self.np_random = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale

    def reset(self) -> None:
        super().reset()
        dim = self.env.feature_dim
        self.b_vec = np.zeros(dim)
        self.inv_A = np.eye(dim) / self.ucb_regularizer
        self.A = np.zeros_like(self.inv_A)
        self.theta = np.zeros(dim)
        self.new_b_vec = np.zeros(dim)
        self.new_inv_A = np.eye(dim) / self.ucb_regularizer
        self.new_theta = np.zeros(dim)
        self.param_bound = 1
        self.features_bound = 1

    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.env.action_space.n
        dim = features.shape[1]
        # beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2
        #                                               *self.t/self.ucb_regularizer)/self.delta)) \
        #        + self.param_bound * np.sqrt(self.ucb_regularizer)
        beta = np.sqrt(np.log(self.t+1))

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
        self.features_bound = max(self.features_bound, np.linalg.norm(v, 2).item())
        self.writer.add_scalar('features_bound', self.features_bound, self.t)

        self.A += np.outer(v,v)
        self.new_b_vec = self.new_b_vec + v * reward
        self.new_inv_A, den = inv_sherman_morrison(v, self.new_inv_A)
        # self.A_logdet += np.log(den)
        self.new_theta = self.inv_A @ self.b_vec
        self.param_bound = np.linalg.norm(self.theta, 2).item()
        self.writer.add_scalar('param_bound', self.param_bound, self.t)

        self.inv_A = self.new_inv_A
        self.theta = self.new_theta
        self.b_vec = self.new_b_vec

    def train(self) -> float:
        if self.t % self.update_every_n_steps == 0:
            self.inv_A = self.new_inv_A
            self.theta = self.new_theta
            self.b_vec = self.new_b_vec

            min_eig = np.linalg.eigvalsh(self.A/(self.t+1)).min() / self.features_bound
            self.writer.add_scalar('min_eig', min_eig, self.t)
        
        return 0

