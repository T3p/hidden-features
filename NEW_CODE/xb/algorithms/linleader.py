import numpy as np
from .linucb import LinUCB, BatchedLinUCB
import numbers


class LinLEADER:

    def __init__(self, reps,
        reg_val, noise_std,
        features_bound,
        param_bound,
        bonus_scale=1., delta=0.01, adaptive_ci=True, random_state=0
    ) -> None:
        # check correctness of inputs
        n_reps = len(reps)
        if isinstance(features_bound, numbers.Number):
            features_bound = [features_bound] * n_reps
        if isinstance(param_bound, numbers.Number):
            param_bound = [param_bound] * n_reps
        if isinstance(bonus_scale, numbers.Number):
            bonus_scale = [bonus_scale] * n_reps
        assert len(param_bound) == len(features_bound) and len(param_bound) == n_reps and len(bonus_scale) == n_reps
        # create base algorithms
        self.algs = []
        for i in range(n_reps):
            self.algs.append(
                LinUCB(
                    rep=reps[i], reg_val=reg_val, noise_std=noise_std,
                    features_bound=features_bound[i], param_bound=param_bound[i],
                    bonus_scale=bonus_scale[i], delta=delta, adaptive_ci=adaptive_ci, random_state=random_state
                )
            )
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def reset(self):
        M = len(self.algs)
        self.active_reps = list(range(M))
        for el in self.algs:
            el.reset()
        self.n_updates = 0
    
    def update(self, context, action, reward):
        self.n_updates += 1
        for i in self.active_reps:
            self.algs[i].update(context, action, reward)

    def compute_scores(self, context, available_actions):
        M_active, n_actions = len(self.active_reps), len(available_actions)
        scores = np.zeros((M_active, n_actions))
        for i, rep_id in enumerate(self.active_reps):
            scores[i] = self.algs[rep_id].compute_scores(context, available_actions)
        return scores

    def action(self, context, available_actions):
        scores = self.compute_scores(context, available_actions)
        action = np.argmax(np.min(scores,axis=0))
        return available_actions[action]


class BatchedLinLEADER(LinLEADER):

    def __init__(self, reps,
        reg_val, noise_std,
        features_bound,
        param_bound, batch_type="det", batch_param=1,
        bonus_scale=1., delta=0.01, adaptive_ci=True, random_state=0
    ) -> None:
        # check correctness of inputs
        n_reps = len(reps)
        if isinstance(features_bound, numbers.Number):
            features_bound = [features_bound] * n_reps
        if isinstance(param_bound, numbers.Number):
            param_bound = [param_bound] * n_reps
        if isinstance(bonus_scale, numbers.Number):
            bonus_scale = [bonus_scale] * n_reps
        if not isinstance(batch_type, list):
            batch_type = [batch_type] * n_reps
            batch_param = [batch_param] * n_reps
        assert len(param_bound) == len(features_bound) and len(param_bound) == n_reps and len(bonus_scale) == n_reps
        # create base algorithms
        self.algs = []
        for i in range(n_reps):
            self.algs.append(
                BatchedLinUCB(
                    rep=reps[i], reg_val=reg_val, noise_std=noise_std,
                    features_bound=features_bound[i], param_bound=param_bound[i],
                    bonus_scale=bonus_scale[i], delta=delta, adaptive_ci=adaptive_ci, random_state=random_state,
                    batch_param=batch_param[i], batch_type=batch_type[i]
                )
            )


class LinLEADERElim(LinLEADER):

    def __init__(self, reps,
        reg_val, noise_std,
        features_bound,
        param_bound,
        bonus_scale=1., delta=0.01, adaptive_ci=True, random_state=0,
        check_elim_condition_every=1
    ) -> None:
        super().__init__(
            reps=reps, reg_val=reg_val, noise_std=noise_std,
            features_bound=features_bound, param_bound=param_bound,
            bonus_scale=bonus_scale, delta=delta, adaptive_ci=adaptive_ci, random_state=random_state
        )
        self.check_elim_condition_every=check_elim_condition_every
        self.eliminate_in_update = True

    def reset(self):
        M = len(self.algs)
        self.active_reps = list(range(M))
        self.SSE_A = np.zeros(M)
        self.SSE_B = []
        self.SSE_C = []
        self.SSE_D = []
        for el in self.algs:
            el.reset()
            d = el.inv_A.shape[0]
            self.SSE_B.append(np.zeros(d))
            self.SSE_C.append(np.zeros(d))
            k = int((d * (d-1)) / 2)
            self.SSE_D.append(np.zeros(k))
        self.t = 1

    
    def update(self, context, action, reward):
        self.t += 1
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
        for i in self.active_reps:
            self.algs[i].update(context, action, reward)
            v = self.algs[i].rep.get_features(context, action)
            d = len(v)
            self.SSE_A[i] += reward * reward
            self.SSE_B[i] = self.SSE_B[i] - 2 * reward * v
            self.SSE_C[i] = self.SSE_C[i] + v**2
            idx = 0
            for k in range(d):
                for j in range(k+1, d):
                    self.SSE_D[i][idx] += 2*v[k]*v[j]
                    idx += 1
        if self.t % self.check_elim_condition_every == 0 and self.eliminate_in_update:
            # compute mses and evaluate elimination condition
            mse, min_mse_plusoffset = self.compute_mses()
            new_active_reps = []
            for i in self.active_reps:
                if mse[i] <= min_mse_plusoffset:
                    new_active_reps.append(i)
                else:
                    print(f"t({self.t}): elim {i} since {mse[i]} > {min_mse_plusoffset}")
            if len(new_active_reps) != len(self.active_reps):
                print(f"new active set: {new_active_reps}")
            self.active_reps = new_active_reps

    def compute_mses(self):
        M = len(self.algs)
        mse = -np.inf*np.ones(M)
        min_val = np.inf
        for i in self.active_reps:
            model = self.algs[i]
            L = model.features_bound
            S = model.param_bound
            d = model.inv_A.shape[0]
            tt = self.t
            offset = (24 * np.log(2 * tt) + 8* np.log(M * M / model.delta) + 8 * d * np.log(12 * L * S * tt) + 1) / tt
            sse = self.SSE_A[i]
            idx = 0 
            for k in range(d):
                sse += (self.SSE_B[i][k] + self.SSE_C[i][k]*model.theta[k]) * model.theta[k]
                for j in range(k+1,d):
                    sse += self.SSE_D[i][idx] * model.theta[k] * model.theta[j]
                    idx += 1
            mse[i] = sse / tt
            min_val = min(min_val, mse[i] + offset)

        return mse, min_val
        