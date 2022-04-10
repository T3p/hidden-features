import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter

class LEADER:
  
    def __init__(
        self, env, representations, reg_val, noise_std,
        features_bounds,
        param_bounds, delta=0.01, random_state=0,
        check_elim_condition_every=1,
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
        self.check_elim_condition_every = check_elim_condition_every
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
        M = len(self.reps)

        active_reps = list(range(M))

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
        b_vec = []
        A_logdet = []
        theta = []
        for i in range(M):
            dim = self.reps[i].features_dim()
            inv_A.append(np.eye(dim) / self.reg_val)
            b_vec.append(np.zeros(dim))
            A_logdet.append(np.log(self.reg_val) * dim)
            theta.append(inv_A[i] @ b_vec[i])
        
        for t in range(horizon):
            context = self.env.sample_context()
            avail_actions = self.env.get_available_actions()

            # compute UCBs of active representations
            M_active, n_actions = len(active_reps), len(avail_actions)
            scores = np.zeros((M_active, n_actions))
            for j, rep_id in enumerate(active_reps):
                dim = self.reps[j].features_dim()
                val = A_logdet[rep_id] - dim * np.log(self.reg_val) - 2 * np.log(self.delta)
                beta = self.noise_std * np.sqrt(val) + self.param_bound[rep_id] * np.sqrt(self.reg_val)

                for i, a in enumerate(avail_actions):
                    v = self.reps[rep_id].get_features(context, a)
                    tie_breaking_noise = self.rng.randn() * 1e-15
                    norm_val = v.dot(inv_A[rep_id].dot(v))
                    bonus = beta * np.sqrt(norm_val)
                    scores[j, i] = v.dot(theta[rep_id]) + bonus + tie_breaking_noise
            
            # select and execute action
            action = np.argmax(np.min(scores, axis=0))
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
            for i in active_reps:
                v = self.reps[i].get_features(context, action)
                d = len(v)
                b_vec[i] += v * reward
                den = 1. + v.dot(inv_A[i].dot(v))
                outp = np.outer(v,v)
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

            # eliminate?
            if t % self.check_elim_condition_every == 0:
                # compute MSEs
                mse = -np.inf*np.ones(M)
                min_mse_plusoffset = np.inf
                for i in active_reps:
                    L = self.features_bound[i]
                    S = self.param_bound[i]
                    d = inv_A[i].shape[0]
                    tt = t+1
                    offset = (24 * np.log(2 * tt) + 8* np.log(M * M / self.delta) + 8 * d * np.log(12 * L * S * tt) + 1) / tt
                    sse = SSE_A[i]
                    idx = 0 
                    for k in range(d):
                        sse += (SSE_B[i][k] + SSE_C[i][k] * theta[i][k]) * theta[i][k]
                        for j in range(k+1,d):
                            sse += SSE_D[i][idx] * theta[i][k] * theta[i][j]
                            idx += 1
                    mse[i] = sse / tt
                    min_mse_plusoffset = min(min_mse_plusoffset, mse[i] + offset)

                # evaluate elimination condition
                new_active_reps = []
                for i in active_reps:
                    if mse[i] <= min_mse_plusoffset:
                        new_active_reps.append(i)
                    else:
                        self.logger.info(f"t({t}): elim {i} since {mse[i]} > {min_mse_plusoffset}")
                if len(new_active_reps) != len(active_reps):
                    self.logger.info(f"new active set: {new_active_reps}")
                active_reps = new_active_reps

            # regret computation
            instant_reward[t] = self.env.expected_reward(action)
            best_reward[t] = self.env.best_reward()
            reg += best_reward[t] - instant_reward[t]
            writer.add_scalar('expected regret', reg, t)
        writer.close()
        return {"regret": np.cumsum(best_reward - instant_reward)}
