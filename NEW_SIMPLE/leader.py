import numpy as np
from typing import List, Tuple
from linearenv import LinearRepresentation

class Leader:
  
    def __init__(
        self, env, representations: List[LinearRepresentation], reg_val: float, noise_std: float,
        features_bounds: List[float],
        param_bounds: List[float], 
        delta:float=0.01, random_state:int=0, adaptive_ci:bool=True
    ) -> None:
        self.env = env
        self.reps = representations
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bounds = features_bounds
        self.param_bounds=param_bounds
        self.delta = delta
        self.random_state = random_state
        self.adaptive_ci = adaptive_ci
        self.rng = np.random.RandomState(random_state)

    def reset(self, horizon: int) -> None:
        self.instant_reward = np.zeros(horizon)
        self.best_reward = np.zeros(horizon)
        M = len(self.reps)
        self.inv_A = []
        self.b_vec = []
        self.A_logdet = []
        self.theta = []
        for i in range(M):
            dim = self.reps[i].features_dim()
            self.inv_A.append(np.eye(dim) / self.reg_val)
            self.b_vec.append(np.zeros(dim))
            self.A_logdet.append(np.log(self.reg_val) * dim)
            self.theta.append(self.inv_A[i] @ self.b_vec[i])
        self.t = 1

    def compute_scores(self, context, available_actions) -> np.ndarray:
        M = len(self.reps)
        scores = np.zeros((M, len(available_actions)))
        for j in range(M):
            dim = self.rep.features_dim()
        
            if self.adaptive_ci:
                val = self.A_logdet[j] - dim * np.log(self.reg_val) - 2 * np.log(self.delta)
                beta = self.noise_std * np.sqrt(val) + self.param_bounds[j] * np.sqrt(self.reg_val)
            else:
                beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bounds[j]*self.features_bound*self.t/self.reg_val)/self.delta)) + self.param_bounds[j] * np.sqrt(self.reg_val)
        
            for i, a in enumerate(available_actions):
                v = self.reps[j].get_features(context, a)
                tie_breaking_noise = self.rng.randn() * 1e-15
                norm_val = v.dot(self.inv_A[j].dot(v))
                bonus = beta * np.sqrt(norm_val)
                scores[j,i] = v.dot(self.theta[j]) + self.bonus_scale * bonus + tie_breaking_noise
        return scores

    def action(self, context, available_actions) -> np.ndarray:
        scores = self.compute_scores(context, available_actions)
        action = np.argmax(np.min(scores, axis=0))
        return available_actions[action]

    def update(self, context, action, reward) -> None:
        self.t += 1
        M = len(self.reps)
        for j in range(M):
            v = self.reps[j].get_features(context, action)
            self.b_vec[j] += v * reward
            den = 1. + v.dot(self.inv_A[j].dot(v))
            outp = np.outer(v,v)
            self.inv_A[j] -= self.inv_A[j].dot(outp.dot(self.inv_A[j])) / den
            self.A_logdet[j] += np.log(den)
            self.theta = self.inv_A[j] @ self.b_vec[j]
    
    def _continue(self, horizon: int) -> None:
        """Continue learning from the point where we stopped
        """
        self.instant_reward.resize(horizon)
        self.best_reward.resize(horizon)

    def run(self, horizon: int):

        self._continue()
        
        for t in range(horizon):
            context = self.env.sample_context()
            avail_actions = self.env.get_available_actions()            
            action = self.action(context=context, available_actions=avail_actions)
            reward = self.env.step(action)

            # update
            self.update(context, action, reward)

            # regret computation
            self.instant_reward[self.t] = self.env.expected_reward(action)
            self.best_reward[self.t] = self.env.best_reward()
        
        return {"regret": np.cumsum(self.best_reward - self.instant_reward)}
