import numpy as np
import torch

class TorchLinUCB(torch.nn.Module):

    def __init__(self,
        input_dim,
        reg_val, noise_std,
        features_bound,
        param_bound,
        bonus_scale=1., delta=0.01, adaptive_ci=True, seed=0
    ) -> None:
        self.input_dim = input_dim
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.bonus_scale = bonus_scale
        self.delta = delta
        self.adaptive_ci = adaptive_ci
        self.seed = seed
        self.rng = np.random.RandomState(seed)


    def reset(self):
        dim = self.rep.features_dim()
        self.inv_A = np.eye(dim) / self.reg_val
        self.b_vec = np.zeros(dim)
        self.A_logdet = np.log(self.reg_val) * dim
        self.t = 1
        self.theta = self.inv_A @ self.b_vec
    
    def compute_scores(self, features):
        """
        features: n_actions x dim_features
        """
        dim = features.shape[1]
        if self.adaptive_ci:
            val = self.A_logdet - dim * np.log(self.reg_val) - 2 * np.log(self.delta)
            beta = self.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.reg_val)
        else:
            beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.reg_val)/self.delta)) + self.param_bound * np.sqrt(self.reg_val)
        
        pred = features @ self.theta
        bonus = torch.einsum('...i,...i->...', features @ self.inv_A, features)
        values = pred + beta * bonus

        scores = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            v = features[i]
            tie_breaking_noise = 0
            norm_val = v.dot(self.inv_A.dot(v))
            bonus = beta * np.sqrt(norm_val)
            scores[i] = v.dot(self.theta) + self.bonus_scale * bonus + tie_breaking_noise
        assert np.allclose(scores, values)
        return scores

    def action(self, context, available_actions):
        scores = self.compute_scores(context, available_actions)
        action = np.argmax(scores)
        return available_actions[action]

    def update(self, features, rewards):
        """
        Update the design matrix and response vector given
        n features and the associated rewards
        features: n_steps x dim_features
        rewards: n_steps x reward
        """
        self.t += features.shape[0]
        self.b_vec += torch.sum(features * rewards, dim=0)
        for v in features:
            den = 1. + v.dot(self.inv_A.dot(v))
            self.inv_A -= self.inv_A.dot(torch.outer(v,v).dot(self.inv_A)) / den
            self.A_logdet += np.log(den)
        self.theta = self.inv_A @ self.b_vec