import numpy as np
from linearenv import LinearRepresentation

class LinUCB:
  
    def __init__(
        self, env, representation: LinearRepresentation, reg_val: float, noise_std: float,
        features_bound: float,
        param_bound: float, delta:float=0.01, random_state:int=0, adaptive_ci:bool=True
    ):
        self.env = env
        self.rep = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.delta = delta
        self.random_state = random_state
        self.adaptive_ci = adaptive_ci
        self.rng = np.random.RandomState(random_state)

    def reset(self, horizon: int) -> None:
        self.instant_reward = np.zeros(horizon)
        self.best_reward = np.zeros(horizon)
        dim = self.rep.features_dim()
        self.A = np.zeros((dim,dim))
        self.inv_A = np.eye(dim) / self.reg_val
        self.b_vec = np.zeros(dim)
        self.A_logdet = np.log(self.reg_val) * dim
        self.t = 1
        self.theta = self.inv_A @ self.b_vec

    def compute_scores(self, context, available_actions):
        dim = self.rep.features_dim()
        scores = np.zeros(len(available_actions))
        if self.adaptive_ci:
            val = self.A_logdet - dim * np.log(self.reg_val) - 2 * np.log(self.delta)
            beta = self.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.reg_val)
        else:
            beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.reg_val)/self.delta)) + self.param_bound * np.sqrt(self.reg_val)
        
        for i, a in enumerate(available_actions):
            v = self.rep.get_features(context, a)
            tie_breaking_noise = self.rng.randn() * 1e-15
            norm_val = v.dot(self.inv_A.dot(v))
            bonus = beta * np.sqrt(norm_val)
            scores[i] = v.dot(self.theta) + self.bonus_scale * bonus + tie_breaking_noise
        return scores

    def action(self, context, available_actions) -> np.ndarray:
        scores = self.compute_scores(context, available_actions)
        action = np.argmax(scores)
        return available_actions[action]

    def update(self, context, action, reward) -> None:
        self.t += 1
        v = self.rep.get_features(context, action)
        self.b_vec += v * reward
        den = 1. + v.dot(self.inv_A.dot(v))
        outp = np.outer(v,v)
        self.inv_A -= self.inv_A.dot(outp.dot(self.inv_A)) / den
        self.A += outp
        self.A_logdet += np.log(den)
        self.theta = self.inv_A @ self.b_vec

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
