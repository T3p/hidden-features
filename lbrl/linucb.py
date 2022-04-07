import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

class LinUCB:
  
    def __init__(self, env, representation, reg_val, noise_std,
        features_bound,
        param_bound, delta=0.01, random_state=0, logger=None):
        self.env = env
        self.rep = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.delta = delta
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        
    def run(self, horizon, log_path: str=None):
        if log_path is None:
            log_path = f"tblogs/{type(self).__name__}_{self.env.dataset_name}"
        log_path = log_path
        writer = SummaryWriter(log_path)
        instant_reward = np.zeros(horizon)
        best_reward = np.zeros(horizon)
        reg = 0
        
        dim = self.rep.features_dim()
        inv_A = np.eye(dim) / self.reg_val
        b_vec = np.zeros(dim)
        A_logdet = np.log(self.reg_val) * dim
        theta = inv_A @ b_vec
        for t in range(horizon):
            context = self.env.sample_context()
            avail_actions = self.env.get_available_actions()
            # compute UCBs
            scores = np.zeros(len(avail_actions))
            val = A_logdet - dim * np.log(self.reg_val) - 2 * np.log(self.delta)
            beta = self.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.reg_val)

            for i, a in enumerate(avail_actions):
                v = self.rep.get_features(context, a)
                tie_breaking_noise = self.rng.randn() * 1e-15
                norm_val = v.dot(inv_A.dot(v))
                bonus = beta * np.sqrt(norm_val)
                scores[i] = v.dot(theta) + bonus + tie_breaking_noise
            
            # select and execute action
            action = np.argmax(scores)
            reward = self.env.step(action)

            # update
            v = self.rep.get_features(context, action)
            b_vec += v * reward
            den = 1. + v.dot(inv_A.dot(v))
            outp = np.outer(v,v)
            inv_A -= inv_A.dot(outp.dot(inv_A)) / den
            A_logdet += np.log(den)
            theta = inv_A @ b_vec

            # regret computation
            instant_reward[t] = self.env.expected_reward(action)
            best_reward[t] = self.env.best_reward()
            reg += best_reward[t] - instant_reward[t]
            writer.add_scalar('expected regret', reg, t)
        writer.close()
        return {"regret": np.cumsum(best_reward - instant_reward)}
