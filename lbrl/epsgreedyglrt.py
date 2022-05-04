import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

class EpsGreedyGLRT:
  
    def __init__(self, env, representation, reg_val, noise_std,
        features_bound,
        param_bound, delta=0.01, 
        check_glrt=True,
        random_state=0, logger=None):
        self.env = env
        self.rep = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.delta = delta
        self.check_glrt = check_glrt
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        
    def run(self, horizon, log_path: str=None):
        if log_path is None:
            log_path = f"tblogs/{type(self).__name__}"
        log_path = log_path
        writer = SummaryWriter(log_path)
        instant_reward = np.zeros(horizon)
        best_reward = np.zeros(horizon)
        reg = 0
        
        dim = self.rep.features_dim()
        inv_A = np.eye(dim) / self.reg_val
        b_vec = np.zeros(dim)
        theta = inv_A @ b_vec
        glrt_values = np.zeros(self.env.n_actions)
        feat_x = np.zeros((self.env.n_actions, dim))
        for t in range(horizon):
            context = self.env.sample_context()
            avail_actions = self.env.get_available_actions()
            #GLRT
            tt = t + 1
            # epsilon = 1. / np.sqrt(tt)
            epsilon = 1. / np.cbrt(tt)
            for i, a in enumerate(avail_actions):
                v = self.rep.get_features(context, a)
                feat_x[i] = v
            rew_hat = feat_x @ theta
            amax = np.argmax(rew_hat)
            for i in range(len(avail_actions)):
                if i == amax:
                    glrt_values[i] = np.inf
                else:
                    xx = feat_x[amax] - feat_x[i]
                    val = xx.dot(inv_A @ xx)
                    glrt_values[i] = (rew_hat[amax] - rew_hat[i])**2 / (2*(val))
            val = 2 * np.log(1./self.delta) + dim * np.log(1 + 2*tt*self.features_bound/(self.reg_val*dim))
            betasq = self.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.reg_val)
            betasq *= betasq
            glrt_in = False
            glrt_minval = np.min(glrt_values)
            writer.add_scalars('glrt test', {"minval": glrt_minval, "threshold": betasq}, t)

            if self.check_glrt and glrt_minval > betasq:
                glrt_in = True
                action = amax
            else:
                if self.rng.rand() < epsilon:
                    action = self.rng.choice(len(avail_actions), size=1).item()
                else:
                    action = amax

            # select and execute action
            reward = self.env.step(action)

            # update
            v = self.rep.get_features(context, action)
            b_vec += v * reward
            den = 1. + v.dot(inv_A.dot(v))
            outp = np.outer(v,v)
            inv_A -= inv_A.dot(outp.dot(inv_A)) / den
            theta = inv_A @ b_vec

            # regret computation
            instant_reward[t] = self.env.expected_reward(action)
            best_reward[t] = self.env.best_reward()
            reg += best_reward[t] - instant_reward[t]
            writer.add_scalar('expected regret', reg, t)
            writer.add_scalar('epsilon', epsilon, t)
            writer.add_scalar('glrt active', 1 if glrt_in else 0, t)
        writer.close()
        return {"regret": np.cumsum(best_reward - instant_reward)}
