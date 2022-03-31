import numpy as np
from typing import Optional, Any, List
from ..batched.templates import XBModule
from ...envs.hlsutils import optimal_features, min_eig_outer
import torch

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + np.dot(u.T, Au)
    A_inv -= np.outer(Au, Au) / (den)
    return A_inv, den

class LeaderSelectElim(XBModule):

    def __init__(
        self,
        env: Any,
        reps: List[Any],
        seed: Optional[int] = 0, 
        update_every_n_steps: Optional[int] = 1,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.
    ) -> None:
        super().__init__(env, None, None, None, None, None, None, 0, seed, None, update_every_n_steps)
        self.reps = reps
        self.np_random = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale

    def reset(self) -> None:
        super().reset()
        M = len(self.reps)

        self.SSE_A = np.zeros(M)
        self.SSE_B = []
        self.SSE_C = []
        self.SSE_D = []
        self.b_vec = []
        self.inv_A = []
        self.A = []
        self.theta = []
        self.param_bound = []
        self.features_bound = []

        for j in range(M):
            dim = self.reps[j].embedding_dim
            self.b_vec.append(torch.zeros(dim))
            self.inv_A.append(torch.eye(dim) / self.ucb_regularizer)
            self.A.append(torch.zeros((dim, dim)))
            self.theta.append(torch.zeros(dim))
            self.param_bound.append(1)
            self.features_bound.append(1)

            self.SSE_B.append(torch.zeros(dim))
            self.SSE_C.append(torch.zeros(dim))
            k = int((dim * (dim-1)) / 2)
            self.SSE_D.append(torch.zeros(k))

    
    @torch.no_grad()
    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.env.action_space.n
        nreps = len(self.reps)

        min_eigs = np.zeros(nreps)
        for i in range(nreps):
            dm = self.A
            eigs, _ = np.linalg.eig(dm)
            min_eigs[i] = np.real(eigs).min()
        mse, min_mse_plusoffset = self.compute_mses()
        cond = mse > min_mse_plusoffset
        # if np.any(cond):
        #     print(f"reps {np.arange(M)[cond]} do not satisfy the constraint ({mse[cond]} > {min_mse_plusoffset})")
        # if condition is False, we set the maximum value
        # i.e. the representation will not be selected
        value = min_eigs + cond * -np.finfo(float).max 
        self.selected_rep = np.argmax(value)
        
        # UCB Step
        xt = torch.FloatTensor(features).to(self.device)
        net_features = self.model.embedding(xt)
        features = self.reps[self.selected_rep]
        # dim = features.shape[1]
        # beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2
        #                                               *self.t/self.ucb_regularizer)/self.delta)) \
        #        + self.param_bound * np.sqrt(self.ucb_regularizer)
        #beta=self.noise_std * np.sqrt(-2 * np.log(np.sqrt(np.linalg.det(self.inv_A)) * self.ucb_regularizer**(dim / 2) * self.delta )) + np.sqrt(self.ucb_regularizer) * self.param_bound
        beta = np.sqrt(np.log(self.t+1))
        #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
        bonus = ((net_features @ self.inv_A)*net_features).sum(axis=1)
        bonus = self.bonus_scale * beta * torch.sqrt(bonus)
        ucb = net_features @ self.theta + bonus
        action = torch.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        assert 0 <= action < self.env.action_space.n, ucb


    @torch.no_grad()
    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        nreps = len(self.reps)
        for i in range(nreps):
            xt = torch.FloatTensor(features.reshape(1,-1)).to(self.device)
            v = self.reps[i].embedding(xt).squeeze()
            d = self.A.shape[0]

            # self.features_bound = max(self.features_bound, torch.norm(v, p=2).cpu().item())
            # self.writer.add_scalar('features_bound', self.features_bound, self.t)

            self.A[i] += torch.outer(v.ravel(),v.ravel())
            self.b_vec[i] = self.b_vec[i] + v * reward
            self.inv_A[i], den = inv_sherman_morrison(v, self.inv_A[i])
            # self.A_logdet[i] += np.log(den)
            self.theta[i] = self.inv_A[i] @ self.b_vec[i]

            self.param_bound[i] = max(self.param_bound[i], torch.linalg.norm(self.theta[i], 2).item())
            self.features_bound[i] = max(self.features_bound[i], torch.norm(v, p=2, dim=1).max().cpu())

            # update MSE terms
            self.SSE_A[i] += reward * reward
            self.SSE_B[i] = self.SSE_B[i] - 2 * reward * v
            self.SSE_C[i] = self.SSE_C[i] + v**2
            idx = 0
            for k in range(d):
                for j in range(k+1, d):
                    self.SSE_D[i][idx] += 2*v[k]*v[j]
                    idx += 1

    def train(self) -> float:
        return 0

    def compute_mses(self):
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
        mse = -np.inf*np.ones(M)
        min_val = np.inf
        for i in self.active_reps:
            L = self.features_bound[i]
            S = self.param_bound[i]
            d = self.inv_A[i].shape[0]
            tt = self.t
            offset = (24 * np.log(2 * tt) + 8* np.log(M * M / self.delta) + 8 * d * np.log(12 * L * S * tt) + 1) / tt
            sse = self.SSE_A[i]
            idx = 0 
            theta_i = self.theta[i]
            for k in range(d):
                sse += (self.SSE_B[i][k] + self.SSE_C[i][k]*theta_i[k]) * theta_i[k]
                for j in range(k+1,d):
                    sse += self.SSE_D[i][idx] * theta_i[k] * theta_i[j]
                    idx += 1
            mse[i] = sse / tt
            min_val = min(min_val, mse[i] + offset)

        return mse, min_val
        