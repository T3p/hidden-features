import numpy as np


class Runner():

    def __init__(self, env, algo, T_max) -> None:
        self.env = env
        self.algo = algo
        self.T_max = T_max
    
    def reset(self):
        self.t = 1
        self.env.reset()
        self.algo.reset()
        self.instant_reward = np.zeros(self.T_max)
        self.best_reward = np.zeros(self.T_max)

    def __call__(self):
        self.instant_reward.resize(self.T_max)
        self.best_reward.resize(self.T_max)
        while (self.t < self.T_max):
            context = self.env.sample_context()
            avail_action_idxs = self.env.get_available_actions()
            action = self.algo.action(context, avail_action_idxs)
            reward = self.env.step(action)
            self.algo.update(context, action, reward)

            self.instant_reward[self.t] = reward
            self.best_reward[self.t] = self.env.best_reward()

            self.t += 1
        return {"regret": np.cumsum(self.best_reward - self.instant_reward)}