import numpy as np

class LinUCB:

    def __init__(self, rep,
        reg_val, noise_std,
        features_bound,
        param_bound,
        bonus_scale=1., delta=0.01, adaptive_ci=True, random_state=0
    ) -> None:
        self.rep = rep
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.bonus_scale = bonus_scale
        self.delta = delta
        self.adaptive_ci = adaptive_ci
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def reset(self):
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

    def action(self, context, available_actions):
        scores = self.compute_scores(context, available_actions)
        action = np.argmax(scores)
        return available_actions[action]

    def update(self, context, action, reward):
        self.t += 1
        v = self.rep.get_features(context, action)
        self.b_vec += v * reward
        den = 1. + v.dot(self.inv_A.dot(v))
        outp = np.outer(v,v)
        self.inv_A -= self.inv_A.dot(outp.dot(self.inv_A)) / den
        self.A += outp
        self.A_logdet += np.log(den)
        self.theta = self.inv_A @ self.b_vec


class BatchedLinUCB(LinUCB):

    def __init__(self, rep,
        reg_val, noise_std,
        features_bound,
        param_bound, batch_type="det", batch_param=1,
        bonus_scale=1., delta=0.01, adaptive_ci=True, random_state=0
    ) -> None:
        assert batch_type in ["det", "fix"]
        self.batch_param = batch_param
        self.batch_type = batch_type
        super().__init__(
            rep, reg_val, noise_std, 
            features_bound, param_bound, bonus_scale, 
            delta, adaptive_ci, random_state)

    def reset(self):
        dim = self.rep.features_dim()
        self.inv_A = np.eye(dim) / self.reg_val
        self.b_vec = np.zeros(dim)
        self.A_logdet = np.log(self.reg_val) * dim
        self.t = 1
        self.new_inv_A = np.eye(dim) / self.reg_val
        self.new_b_vec = np.zeros(dim)
        self.new_A_logdet = np.log(self.reg_val) * dim
        self.theta = self.inv_A @ self.b_vec
    
    def update(self, context, action, reward):
        self.t += 1
        v = self.rep.get_features(context, action)
        self.new_b_vec += v * reward
        den = 1. + v.dot(self.new_inv_A.dot(v))
        self.new_inv_A -= self.new_inv_A.dot(np.outer(v,v).dot(self.new_inv_A)) / den
        self.new_A_logdet += np.log(den)

        update = False
        if self.batch_type == "det":
            if self.new_A_logdet > np.log(1+self.batch_param) + self.A_logdet:
                update = True
        else:
            if self.t % int(self.batch_param) == 0:
                update = True
        if update:
            self.A_logdet = self.new_A_logdet
            self.inv_A = self.new_inv_A.copy()
            self.b_vec = self.new_b_vec.copy()
            self.theta = self.inv_A @ self.b_vec