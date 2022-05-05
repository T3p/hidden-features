import numpy as np
import logging
from .replaybuffer import SimpleBuffer
from torch.utils.tensorboard import SummaryWriter
import pdb

class LEADERSelect:

    MINEIG=0
    MINEIG_NORM=1
    AVG_QUAD=2
    AVG_QUAD_NORM=3
  
    def __init__(
        self, env, representations, reg_val, noise_std,
        features_bounds, select_method,
        param_bounds, delta=0.01, random_state=0,
        logger=None
      ):
        self.env = env
        self.reps = representations
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.features_bound = features_bounds
        self.param_bound = param_bounds
        self.delta = delta
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.select_method = select_method
        assert select_method in [self.MINEIG, self.MINEIG_NORM, self.AVG_QUAD, self.AVG_QUAD_NORM]
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
        M = len(self.reps)

        active_reps = list(range(M))
        hist_mineig = []
        hist_mineig_norm = []
        hist_rep_scores = []
        hist_selected_rep = []
        hist_active_reps = []
        hist_mse = []
        hist_time_updates = []

        update_time = 1
        selected_rep = 0

        # prepare structure for SSE
        SSE_A = np.zeros(M)
        SSE_B = []
        SSE_C = []
        SSE_D = []
        for el in self.reps:
            d = el.features_dim()
            SSE_B.append(np.zeros(d))
            SSE_C.append(np.zeros(d))
            k = int((d * (d-1)) / 2)
            SSE_D.append(np.zeros(k))

        # prepare structure for LinUCB
        reg = 0
        inv_A = []
        Amtx = []
        b_vec = []
        A_logdet = []
        theta = []
        for i in range(M):
            dim = self.reps[i].features_dim()
            inv_A.append(np.eye(dim) / self.reg_val)
            Amtx.append(self.reg_val * np.eye(dim))
            b_vec.append(np.zeros(dim))
            A_logdet.append(np.log(self.reg_val) * dim)
            theta.append(inv_A[i] @ b_vec[i])

        if self.select_method in [self.AVG_QUAD,self.AVG_QUAD_NORM]:
            buffer = SimpleBuffer(capacity=horizon)
        
        for t in range(horizon):
            context_id = self.env.sample_context()
            avail_actions = self.env.get_available_actions()


            # select representation
            M_active, n_actions = len(active_reps), len(avail_actions)
            # if t % self.recompute_every == 0:
            if t == update_time:
                update_time *= 2
                hist_time_updates.append(t)
                min_eigs = np.zeros(M_active)
                min_eigs_norm = np.zeros(M_active)
                rep_scores = None
                if self.select_method in [self.MINEIG_NORM, self.MINEIG]:
                # compute min eigs
                    for i, idx in enumerate(active_reps):
                        dm = Amtx[idx]
                        eigs, _ = np.linalg.eig(dm)
                        assert np.isclose(np.imag(eigs).min(), 0)
                        min_eigs[i] = np.real(eigs).min()
                        min_eigs_norm[i] = min_eigs[i] / (self.features_bound[idx]**2)
                    if self.select_method == self.MINEIG:
                        rep_scores = min_eigs
                    else:
                        rep_scores = min_eigs_norm
                else:
                    # \min_\phi \sum_t \sum_a phi(x_t, a) V phi(x_t,a)
                    rep_scores = np.zeros(M_active)
                    if len(buffer) > 0:
                        obs_context_idxs, _ = buffer.get_all()
                        for i, idx in enumerate(active_reps):
                            Lsq = self.features_bound[idx]**2
                            for cidx in obs_context_idxs:
                                for a in range(n_actions):
                                    phi = self.reps[idx].get_features(cidx, a).squeeze()
                                    if self.select_method == self.AVG_QUAD_NORM:
                                        rep_scores[i] += phi.dot(Amtx[idx] @ phi) / Lsq
                                    else:
                                        rep_scores[i] += phi.dot(Amtx[idx] @ phi)

                    buffer.append((context_id, None))
                
                # compute MSEs
                mse = -np.inf*np.ones(M_active)
                min_mse_plusoffset = np.inf
                for i, rep_id in enumerate(active_reps):
                    L = self.features_bound[rep_id]
                    S = self.param_bound[rep_id]
                    d = inv_A[rep_id].shape[0]
                    tt = t+1
                    offset = (24 * np.log(2 * tt) + 8* np.log(M * M / self.delta) + 8 * d * np.log(12 * L * S * tt) + 1) / tt
                    sse = SSE_A[rep_id]
                    idx = 0 
                    for k in range(d):
                        sse += (SSE_B[rep_id][k] + SSE_C[rep_id][k] * theta[rep_id][k]) * theta[rep_id][k]
                        for j in range(k+1,d):
                            sse += SSE_D[rep_id][idx] * theta[rep_id][k] * theta[rep_id][j]
                            idx += 1
                    mse[i] = sse / tt
                    min_mse_plusoffset = min(min_mse_plusoffset, mse[i] + offset)

                log_eigs = np.zeros(M) - 1
                log_eigs[active_reps] = min_eigs
                log_rep_scores = np.zeros(M) - 99
                log_rep_scores[active_reps] = rep_scores
                log_mineig_norm = np.zeros(M) - 1
                log_mineig_norm[active_reps] = min_eigs_norm
                hist_mineig.append(log_eigs)
                hist_rep_scores.append(log_rep_scores)
                hist_mineig_norm.append(log_mineig_norm)
                log_mse = np.zeros(M) - 1
                log_mse[active_reps] = mse
                hist_mse.append(log_mse)

                writer.add_scalars('min_eig_norm', {f"rep{i}": log_mineig_norm[i] for i in range(M)}, t)
                writer.add_scalars('min_eig', {f"rep{i}": log_eigs[i] for i in range(M)}, t)
                writer.add_scalars('rep_score', {f"rep{i}": log_rep_scores[i] for i in range(M)}, t)
                writer.add_scalars('mse', {f"rep{i}": log_mse[i] for i in range(M)}, t)

                cond = mse > min_mse_plusoffset

                value = rep_scores - cond * np.finfo(float).max
                winners = np.argwhere(value == value.max()).flatten().tolist()
                selected_rep = self.rng.choice(winners)
                selected_rep = active_reps[selected_rep]
                writer.add_scalar('selected rep', selected_rep, t)
                hist_selected_rep.append(selected_rep)
                
                
                # evaluate elimination condition
                new_active_reps = []
                for i, rep_idx in enumerate(active_reps):
                    if mse[i] <= min_mse_plusoffset:
                        new_active_reps.append(rep_idx)
                    else:
                        self.logger.info(f"t({t}): elim {rep_idx} since {mse[i]} > {min_mse_plusoffset}")
                if len(new_active_reps) != len(active_reps):
                    self.logger.info(f"new active set: {new_active_reps}")
                active_reps = new_active_reps
                hist_active_reps.append(active_reps)
                

            # compute UCBs of selected representation
            ucbs = np.zeros(n_actions)
            dim = self.reps[selected_rep].features_dim()
            val = A_logdet[selected_rep] - dim * np.log(self.reg_val) - 2 * np.log(self.delta)
            beta = self.noise_std * np.sqrt(val) + self.param_bound[selected_rep] * np.sqrt(self.reg_val)

            for i, a in enumerate(avail_actions):
                v = self.reps[selected_rep].get_features(context_id, a)
                tie_breaking_noise = self.rng.randn() * 1e-15
                norm_val = v.dot(inv_A[selected_rep].dot(v))
                bonus = beta * np.sqrt(norm_val)
                ucbs[i] = v.dot(theta[selected_rep]) + bonus + tie_breaking_noise
            
            # select and execute action
            action = np.argmax(ucbs)
            reward = self.env.step(action)

            # update
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
            for i in range(M):
                v = self.reps[i].get_features(context_id, action)
                d = len(v)
                b_vec[i] += v * reward
                den = 1. + v.dot(inv_A[i].dot(v))
                outp = np.outer(v,v)
                Amtx[i] += outp
                inv_A[i] -= inv_A[i].dot(outp.dot(inv_A[i])) / den
                A_logdet[i] += np.log(den)
                theta[i] = inv_A[i] @ b_vec[i]
                #update SSE
                SSE_A[i] += reward * reward
                SSE_B[i] = SSE_B[i] - 2 * reward * v
                SSE_C[i] = SSE_C[i] + v**2
                idx = 0
                for k in range(d):
                    for j in range(k+1, d):
                        SSE_D[i][idx] += 2*v[k]*v[j]
                        idx += 1

            # regret computation
            instant_reward[t] = self.env.expected_reward(action)
            best_reward[t] = self.env.best_reward()
            reg += best_reward[t] - instant_reward[t] 
            writer.add_scalar('expected regret', reg, t)
        writer.close()
        return {
            "regret": np.cumsum(best_reward - instant_reward),
            "selected_rep": hist_selected_rep,
            "hist_mineig": hist_mineig,
            "hist_mineig_norm": hist_rep_scores,
            "hist_rep_scores": hist_rep_scores,
            "hist_selected_rep": hist_selected_rep,
            "hist_active_reps": hist_active_reps,
            "hist_mse": hist_mse,
            "hist_time_updates": hist_time_updates
            }
