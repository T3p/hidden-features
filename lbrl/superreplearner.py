import enum
import numpy as np
import logging

import wandb
from .replaybuffer import SimpleBuffer
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List

class SuperRepLearner:

    MINEIG=0
    MINEIG_NORM=1
    AVG_QUAD=2
    AVG_QUAD_NORM=3

    def __init__(
        self, env, representations,
        features_bounds: List,
        param_bounds: List,
        cfg: DictConfig,
    ):
        self.env = env
        self.reps = representations
        self.features_bounds = features_bounds
        self.param_bounds = param_bounds
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.logger = logging.getLogger(__name__)

        # INITIALIZE
        # prepare structure for SSE
        M = len(self.reps)
        assert M > 0
        self.SSE_A = np.zeros(M)
        self.SSE_B = []
        self.SSE_C = []
        self.SSE_D = []
        for el in self.reps:
            d = el.features_dim()
            self.SSE_B.append(np.zeros(d))
            self.SSE_C.append(np.zeros(d))
            k = int((d * (d-1)) / 2)
            self.SSE_D.append(np.zeros(k))
        # prepare structure for LinUCB
        self.inv_A = []
        self.Amtx = []
        self.b_vec = []
        self.A_logdet = []
        self.theta = []
        for i in range(M):
            dim = self.reps[i].features_dim()
            self.inv_A.append(np.eye(dim) / self.cfg.reg_val)
            self.Amtx.append(self.cfg.reg_val * np.eye(dim))
            self.b_vec.append(np.zeros(dim))
            self.A_logdet.append(np.log(self.cfg.reg_val) * dim)
            self.theta.append(self.inv_A[i] @ self.b_vec[i])
        
        self.t = 1
        self.update_time = 1
        self.selected_rep = 0
        self.active_reps = np.ones(M, dtype=int) > 0

    def update(self, context_id, action, reward):
        # SSE = \sum_i (y_i - x_i w)^2
        #     = \sum_i y_i^2 + (x_i w)^2  - 2y_i x_i w
        #     = \sum_i y_i^2 - 2y_i x_i w
        #       + \sum_i [ \sum_j (x_{ij}^2 w_j^2) + 2 \sum_{k}\sum_{j=k+1} x_{ik} x_{ij} w_{ik} w_{ij} ]
        # Let's
        #     A = \sum_i y_i^2
        #     B = \sum_i -2y_i x_i
        #     C = \sum_i x_{ij}^2  (dim d)
        #     D = \sum_i 2 x_{ik} x_{ij} (dim d*(d-1)/2)
        # Then, for any w
        # SSE = A + Bw + \sum_j C_j w_j \sum_{k}\sum_{j=k+1} D_k D_j w_i w_j
        # MSE = SSE / t
        M = len(self.reps)
        for i in range(M):
            # update linear regression
            v = self.reps[i].get_features(context_id, action)
            d = len(v)
            self.b_vec[i] += v * reward
            den = 1. + v.dot(self.inv_A[i].dot(v))
            outp = np.outer(v,v)
            self.Amtx[i] += outp
            self.inv_A[i] -= self.inv_A[i].dot(outp.dot(self.inv_A[i])) / den
            self.A_logdet[i] += np.log(den)
            self.theta[i] = self.inv_A[i] @ self.b_vec[i]
            #update SSE
            self.SSE_A[i] += reward * reward
            self.SSE_B[i] = self.SSE_B[i] - 2 * reward * v
            self.SSE_C[i] = self.SSE_C[i] + v**2
            idx = 0
            for k in range(d):
                for j in range(k+1, d):
                    self.SSE_D[i][idx] += 2*v[k]*v[j]
                    idx += 1

    def compute_mses(self):
        # compute MSEs
        M = len(self.active_reps)
        mse = -np.inf*np.ones(M)
        min_mse_plusoffset = np.inf
        tt = self.t
        for i in range(M):
            if self.active_reps[i] == False:
                mse[i] = np.inf
            else:
                L = self.features_bounds[i]
                S = self.param_bounds[i]
                d = self.inv_A[i].shape[0]
                offset = (24 * np.log(2 * tt) + 8 * np.log(M * M / self.cfg.delta) + 8 * d * np.log(12 * L * S * tt) + 1) / tt
                sse = self.SSE_A[i]
                idx = 0 
                for k in range(d):
                    sse += (self.SSE_B[i][k] + self.SSE_C[i][k] * self.theta[i][k]) * self.theta[i][k]
                    for j in range(k+1,d):
                        sse += self.SSE_D[i][idx] * self.theta[i][k] * self.theta[i][j]
                        idx += 1
                mse[i] = sse / tt
                min_mse_plusoffset = min(min_mse_plusoffset, mse[i] + offset)
        return mse, min_mse_plusoffset

    def glrt(self, context_id, selected_rep):
        """Check GLRT"""
        avail_actions = self.env.get_available_actions()
        dim = self.reps[selected_rep].features_dim()
        feat_x = np.zeros((self.env.n_actions, dim))
        glrt_values = np.zeros(self.env.n_actions)
        for i, a in enumerate(avail_actions):
            v = self.reps[selected_rep].get_features(context_id, a)
            feat_x[i] = v
        rew_hat = feat_x @ self.theta[selected_rep]
        amax = np.argmax(rew_hat)
        for i in range(len(avail_actions)):
            if i == amax:
                glrt_values[i] = np.inf
            else:
                feat_diff = feat_x[amax] - feat_x[i]
                val = feat_diff.dot(self.inv_A[selected_rep] @ feat_diff)
                glrt_values[i] = (rew_hat[amax] - rew_hat[i])**2 / (2*(val))
        val = 2 * np.log(1./self.cfg.delta) + dim * np.log(1 + 2*self.t*self.features_bounds[selected_rep]/(self.cfg.reg_val*dim))
        betasq = self.cfg.noise_std * np.sqrt(val) + self.param_bounds[selected_rep] * np.sqrt(self.cfg.reg_val)
        betasq *= betasq
        glrt_minval = np.min(glrt_values)
        dogreedy = glrt_minval > betasq
        # logging
        if self.cfg.use_tb:
            self.tb_writer.add_scalars('glrt test', {"minval": glrt_minval, "threshold": betasq}, self.t)
        if self.cfg.use_wandb:
            wandb.log({"glrt_minval": glrt_minval, "glrt_threshold": betasq}, step=self.t)
        return dogreedy, amax

    def play_base_algo(self, context_id, selected_rep):
        pass

    def select_rep(self):
        M = len(self.reps)
        rep_scores = np.zeros(M)
        if self.cfg.select_method in [self.MINEIG_NORM, self.MINEIG]:
            for i in range(M):
                if self.active_reps[i]:
                    dm = self.Amtx[i]
                    eigs, _ = np.linalg.eig(dm)
                    assert np.isclose(np.imag(eigs).min(), 0)
                    rep_scores[i] = np.real(eigs).min()
                    if self.cfg.select_method == self.MINEIG_NORM:
                        rep_scores[i] = rep_scores[i] / (self.features_bounds[i]**2)
                else:
                    rep_scores[i] = -np.finfo(float).max
        else:
            # \min_\phi \sum_t \sum_a phi(x_t, a) V phi(x_t,a)
            if len(self.buffer) > 0:
                obs_context_idxs, _ = self.buffer.get_all()
                for i in range(M):
                    if self.active_reps[i]:
                        Lsq = self.features_bounds[i]**2
                        for cidx in obs_context_idxs:
                            for a in range(self.env.n_actions):
                                phi = self.reps[i].get_features(cidx, a).squeeze()
                                if self.cfg.select_method == self.AVG_QUAD_NORM:
                                    rep_scores[i] += phi.dot(self.Amtx[i] @ phi) / Lsq
                                else:
                                    rep_scores[i] += phi.dot(self.Amtx[i] @ phi)
                    else:
                        rep_scores[i] = -np.finfo(float).max
        # logging
        if self.cfg.use_tb:
            self.tb_writer.add_scalars('rep_score', {f"rep{i}": rep_scores[i] for i in range(M)}, self.t)
        if self.cfg.use_wandb:
            wandb.log({f"rep{i}": rep_scores[i] for i in range(M)}, step=self.t)

        mse, min_mse_plusoffset = self.compute_mses()
        elim_rep = mse > min_mse_plusoffset
        rep_scores[elim_rep] = -np.finfo(float).max
        rep_scores[np.logical_not(self.active_reps)] = -np.finfo(float).max
        winners = np.argwhere(rep_scores == rep_scores.max()).flatten().tolist()
        selected_rep = self.rng.choice(winners)
        # evaluate elimination condition
        elim = False
        for i in range(M):
            if mse[i] > min_mse_plusoffset and self.active_reps[i]:
                elim = True
                self.active_reps[i] = False
                self.logger.info(f"t({self.t}): elim {i} since {mse[i]} > {min_mse_plusoffset}")
        if elim:
            self.logger.info(f"new active set: {np.arange(M)[self.active_reps]}")
        return selected_rep
    
    def run(self, horizon: int, throttle: int=100, log_path: str=None) -> None:

        if self.cfg.use_tb:
            if log_path is None:
                log_path = f"tblogs/{type(self).__name__}"
            log_path = log_path
            self.tb_writer = SummaryWriter(log_path)

        if self.cfg.select_method in [self.AVG_QUAD, self.AVG_QUAD_NORM]:
            self.buffer = SimpleBuffer(capacity=horizon)
        instant_reward = np.zeros(horizon)
        best_reward = np.zeros(horizon)

        postfix = {
            'rep': self.selected_rep,
            'expected regret': 0.0,
            'glrt': 0
        }
        reg = 0
        self.selected_rep = 0
        M = len(self.reps)
        hist_selected_rep = []
        hist_time_updates = []
        with tqdm(initial=self.t, total=horizon, postfix=postfix) as pbar:
            while self.t < horizon:
                context_id = self.env.sample_context()
                # avail_actions = self.env.get_available_actions()
                if self.cfg.select_method in [self.AVG_QUAD, self.AVG_QUAD_NORM]:
                    self.buffer.append((context_id, None))

                if self.cfg.check_glrt:
                    doplaygreedy, greedy_action = self.glrt(context_id=context_id, selected_rep=self.selected_rep)
                else:
                    doplaygreedy = False

                if doplaygreedy:
                    action = greedy_action
                else:
                    action = self.play_base_algo(context_id=context_id, selected_rep=self.selected_rep)

                reward = self.env.step(action)
                self.update(context_id=context_id, action=action, reward=reward)

                if M > 1 and self.t == self.update_time:
                    if self.cfg.update_every < 1:
                        self.update_time += 1
                    else:
                        self.update_time *= self.cfg.update_every
                    selected_rep = self.select_rep()
                    if selected_rep is not None:
                        self.selected_rep = selected_rep
                    hist_time_updates.append(self.t)
                    hist_selected_rep.append(self.selected_rep)
                
                # regret computation
                instant_reward[self.t] = self.env.expected_reward(action)
                best_reward[self.t] = self.env.best_reward()
                reg += best_reward[self.t] - instant_reward[self.t] 


                # logging
                postfix['expected regret'] = reg
                postfix['rep'] = self.selected_rep
                postfix['glrt'] = doplaygreedy
                if self.t % throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(throttle)

                if self.cfg.use_tb:
                    self.tb_writer.add_scalar('expected regret', reg, self.t)
                    self.tb_writer.add_scalar('glrt active', 1 if doplaygreedy else 0, self.t)
                if self.cfg.use_wandb:
                    wandb.log({'expected regret': reg, 'glrt active': 1 if doplaygreedy else 0}, step=self.t)
                
                #step
                self.t += 1
        
        if self.cfg.use_tb:
            self.tb_writer.close()

        return {
            "regret": np.cumsum(best_reward - instant_reward),
            "hist_selected_rep": hist_selected_rep,
            "hist_time_updates": hist_time_updates
        }



class SRLEGreedy(SuperRepLearner):

    def play_base_algo(self, context_id, selected_rep):
        self.epsilon = 1. / np.cbrt(self.t)
        if self.cfg.use_tb:
            self.tb_writer.add_scalar('epsilon', self.epsilon, self.t)
        if self.cfg.use_wandb:
            wandb.log({'epsilon', self.epsilon}, step=self.t)
        
        avail_actions = self.env.get_available_actions()
        if self.rng.rand() < self.epsilon:
            action = self.rng.choice(len(avail_actions), size=1).item()
        else:
            rew = []
            for a in avail_actions:
                v = self.reps[selected_rep].get_features(context_id, a).squeeze()
                rew.append(v.dot(self.theta[selected_rep]))
            action = np.argmax(rew)
        return action

class SRLLinUCB(SuperRepLearner):

    def play_base_algo(self, context_id, selected_rep):            
        # compute UCBs of selected representation
        avail_actions = self.env.get_available_actions()
        ucbs = np.zeros(len(avail_actions))
        dim = self.reps[selected_rep].features_dim()
        val = self.A_logdet[selected_rep] - dim * np.log(self.cfg.reg_val) - 2 * np.log(self.cfg.delta)
        beta = self.cfg.noise_std * np.sqrt(val) + self.param_bounds[selected_rep] * np.sqrt(self.cfg.reg_val)

        for i, a in enumerate(avail_actions):
            v = self.reps[selected_rep].get_features(context_id, a)
            tie_breaking_noise = self.rng.randn() * 1e-15
            norm_val = v.dot(self.inv_A[selected_rep].dot(v))
            bonus = beta * np.sqrt(norm_val)
            ucbs[i] = v.dot(self.theta[selected_rep]) + bonus + tie_breaking_noise
        action = np.argmax(ucbs)
        return action


class Leader(SuperRepLearner):

    def select_rep(self):
        assert self.cfg.check_glrt == False
        # eliminate representations
        M = len(self.reps)
        mse, min_mse_plusoffset = self.compute_mses()
        elim = False
        for i in range(M):
            if mse[i] > min_mse_plusoffset and self.active_reps[i]:
                elim = True
                self.active_reps[i] = False
                self.logger.info(f"t({self.t}): elim {i} since {mse[i]} > {min_mse_plusoffset}")
        if elim:
            self.logger.info(f"new active set: {np.arange(M)[self.active_reps]}")
        return None

    def glrt(self, context_id, selected_rep):
        return False, 0

    def play_base_algo(self, context_id, selected_rep):
        avail_actions = self.env.get_available_actions()

        # compute UCBs of active representations
        M = len(self.reps)
        active_reps = np.arange(M)[self.active_reps]
        scores = np.ones((len(active_reps), len(avail_actions))) - np.finfo(float).max
        for j, rep_id in enumerate(active_reps):
            dim = self.reps[rep_id].features_dim()
            val = self.A_logdet[rep_id] - dim * np.log(self.cfg.reg_val) - 2 * np.log(self.cfg.delta)
            beta = self.cfg.noise_std * np.sqrt(val) + self.param_bounds[rep_id] * np.sqrt(self.cfg.reg_val)

            for i, a in enumerate(avail_actions):
                v = self.reps[rep_id].get_features(context_id, a)
                tie_breaking_noise = self.rng.randn() * 1e-15
                norm_val = v.dot(self.inv_A[rep_id].dot(v))
                bonus = beta * np.sqrt(norm_val)
                scores[j, i] = v.dot(self.theta[rep_id]) + bonus + tie_breaking_noise


        # select and execute action
        action = np.argmax(np.min(scores, axis=0))
        return action
