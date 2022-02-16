import numpy as np
from .linleader import LinLEADER, LinLEADERElim


class LinLeaderSelect(LinLEADER):

    def __init__(self, reps,
        reg_val, noise_std,
        features_bound,
        param_bound,
        bonus_scale=1., delta=0.01, 
        adaptive_ci=True, random_state=0,
        time_multiplier_update : int = 2,
        normalize=True
    ) -> None:
        super().__init__(
            reps=reps, reg_val=reg_val, noise_std=noise_std,
            features_bound=features_bound, param_bound=param_bound,
            bonus_scale=bonus_scale, delta=delta, 
            adaptive_ci=adaptive_ci, random_state=random_state,
        )
        self.time_update = 1
        self.time_multiplier_update = time_multiplier_update
        self.selected_rep = None
        self.normalize = normalize
        self.feature_bounds = features_bound

    def reset(self):
        super().reset()
        self.time_update = 1
        self.mineig = []
        self.normalized_rep_scores = []

    def action(self, context, available_actions):
        if self.n_updates % self.time_update == 0:
            self.time_update *= self.time_multiplier_update
            M = len(self.algs)
            # max_eigs_inv = np.zeros(M)
            # for i in range(M):
            #     dm = self.algs[i].inv_A 
            #     eigs, _ = np.linalg.eig(dm)
            #     assert np.isclose(np.imag(eigs).max(), 0)
            #     max_eigs_inv[i] = np.real(eigs).max()
            # self.selected_rep = np.argmin(max_eigs_inv)
            min_eigs = np.zeros(M)
            normalized_rep_scores = np.zeros(M)
            for i in range(M):
                dm = self.algs[i].A
                eigs, _ = np.linalg.eig(dm)
                assert np.isclose(np.imag(eigs).max(), 0)
                min_eigs[i] = np.real(eigs).min()
                normalized_rep_scores[i] = min_eigs[i] / self.feature_bounds[i]**2
            self.mineig.append(min_eigs)
            self.normalized_rep_scores.append(normalized_rep_scores)
            if self.normalize:
                winners = np.argwhere(normalized_rep_scores == 
                                      normalized_rep_scores.max()).flatten().tolist()
            else:
                winners = np.argwhere(min_eigs == min_eigs.max()).flatten().tolist()
            self.selected_rep = self.rng.choice(winners)
        scores = self.algs[self.selected_rep].compute_scores(context, available_actions)
        
        action = np.argmax(scores)
        return available_actions[action]


class LinLeaderElimSelect(LinLEADERElim):

    def __init__(self, reps, reg_val, noise_std, features_bound, param_bound, bonus_scale=1, delta=0.01, adaptive_ci=True, random_state=0, check_elim_condition_every=1) -> None:
        super().__init__(reps, reg_val, noise_std, features_bound, param_bound, bonus_scale=bonus_scale, delta=delta, adaptive_ci=adaptive_ci, random_state=random_state, check_elim_condition_every=check_elim_condition_every)
        self.eliminate_in_update = False
        self.selected_rep = 0

    def action(self, context, available_actions):
        if self.t % self.check_elim_condition_every == 0:
            M = len(self.algs)
            assert len(self.active_reps) == M

            max_eigs_inv = np.zeros(M)
            for i in range(M):
                dm = self.algs[i].inv_A
                eigs, _ = np.linalg.eig(dm)
                assert np.isclose(np.imag(eigs).max(), 0)
                max_eigs_inv[i] = np.real(eigs).max()
            mse, min_mse_plusoffset = self.compute_mses()
            cond = mse > min_mse_plusoffset
            # if np.any(cond):
            #     print(f"reps {np.arange(M)[cond]} do not satisfy the constraint ({mse[cond]} > {min_mse_plusoffset})")
            # if condition is False, we set the maximum value
            # i.e. the representation will not be selected
            value = max_eigs_inv + cond * np.finfo(float).max 
            self.selected_rep = np.argmin(value)

        scores = self.algs[self.selected_rep].compute_scores(context, available_actions)
        action = np.argmax(scores)
        return available_actions[action]   