import numpy as np
from typing import Optional, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from .replaybuffer import SimpleBuffer
from .nnmodel import initialize_weights
import time
import copy
from .. import TORCH_FLOAT


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + torch.dot(u.T, Au)
    A_inv -= torch.outer(Au, Au) / (den)
    return A_inv, den

class NNLeaderInc(nn.Module):

    def __init__(
        self,
        env: Any,
        model: nn.Module,
        device: Optional[str]="cpu",
        batch_size: Optional[int]=256,
        max_updates: Optional[int]=1,
        learning_rate: Optional[float]=0.001,
        weight_decay: Optional[float]=0,
        buffer_capacity: Optional[int]=10000,
        seed: Optional[int]=0,
        update_every: Optional[int] = 100,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.
    ) -> None:
        super().__init__()
        self.env = env
        self.model = model
        self.target_model = copy.deepcopy(self.model).to(device)
        self.target_model.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_updates = max_updates
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_capacity = buffer_capacity
        self.seed = seed
        self.update_every = update_every
        self.unit_vector: Optional[torch.tensor] = None
        self.noise_std = noise_std
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale

    def reset(self) -> None:
        self.t = 0
        self.buffer = SimpleBuffer(capacity=self.buffer_capacity)
        self.instant_reward = np.zeros(1)
        self.best_reward = np.zeros(1)
        self.action_gap = np.zeros(1)
        self.action_history = np.zeros(1, dtype=int)
        self.best_action_history = np.zeros(1, dtype=int)
        self.runtime = np.zeros(1)

        self.batch_counter = 0
        if self.model:
            self.model.to(self.device)

        # TODO: check the following lines: with initialization to 0 the training code is never called
        # self.update_time = 0
        self.update_time = int(2**np.ceil(np.log(self.batch_size) / np.log(self.update_every)) + 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.tot_update = 0 
        dim = self.model.embedding_dim
        self.b_vec = torch.zeros(dim, dtype=torch.float).to(self.device)
        self.inv_A = torch.eye(dim, dtype=torch.float).to(self.device) / self.ucb_regularizer
        self.A = torch.zeros_like(self.inv_A)
        self.theta = torch.zeros(dim, dtype=torch.float).to(self.device)
        self.param_bound = np.sqrt(self.env.feature_dim)
        self.features_bound = np.sqrt(self.env.feature_dim)
    
    @torch.no_grad()
    def play_action(self, features: np.ndarray):
        assert features.shape[0] == self.env.action_space.n
        dim = self.target_model.embedding_dim
        # beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2
        #                                               *self.t/self.ucb_regularizer)/self.delta))\
        #        + self.param_bound * np.sqrt(self.ucb_regularizer)
        beta = np.sqrt(np.log(self.t+1))

        # get features for each action and make it tensor
        xt = torch.tensor(features, dtype=TORCH_FLOAT).to(self.device)
        net_features = self.target_model.embedding(xt)
        #https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v/18542314#18542314
        bonus = ((net_features @ self.inv_A)*net_features).sum(axis=1)
        bonus = self.bonus_scale * beta * torch.sqrt(bonus)
        ucb = net_features @ self.theta + bonus
        action = torch.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        assert 0 <= action < self.env.action_space.n, ucb

        return action

    def _update_after_change_of_target(self):

        #################################################
        # Recompute design matrix and weight
        with torch.no_grad():
            # A = np.eye(dim) * self.ucb_regularizer
            dim = self.target_model.embedding_dim
            self.b_vec = torch.zeros(dim).to(self.device)
            self.inv_A = torch.eye(dim).to(self.device) / self.ucb_regularizer
            self.A = torch.zeros_like(self.inv_A)
            self.features_bound = 0
            features, rewards = self.buffer.get_all()
            features = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
            rewards = torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)

            phi = self.target_model.embedding(features)
            # features
            max_norm = torch.norm(phi, p=2, dim=1).max().cpu()
            self.features_bound = max(self.features_bound, max_norm)
            self.b_vec = self.b_vec + (phi * rewards).sum(dim=0)
            #SM
            for v in phi:
                self.inv_A = inv_sherman_morrison(v.ravel(), self.inv_A)[0]
                self.A += torch.outer(v.ravel(),v.ravel())
            self.theta = self.inv_A @ self.b_vec
            self.param_bound = torch.linalg.norm(self.theta, 2).item()
            # self.writer.add_scalar('param_bound', self.param_bound, self.t)
            # self.writer.add_scalar('features_bound', self.features_bound, self.t)
            # min_eig = torch.linalg.eigvalsh(self.A/(self.t+1)).min() / self.features_bound
            # self.writer.add_scalar('min_eig_empirical_design', min_eig, self.t)

            # # debug metric
            # if hasattr(self.env, 'feature_matrix'):
            #     xx = optimal_features(self.env.feature_matrix, self.env.rewards)
            #     assert len(xx.shape) == 2
            #     xt = torch.FloatTensor(xx).to(self.device)
            #     phi = self.model.embedding(xt).detach().cpu().numpy()
            #     norm_v=np.linalg.norm(phi, ord=2, axis=1).max()
            #     mineig = min_eig_outer(phi, False) / phi.shape[0]
            #     self.writer.add_scalar('min_eig_design_opt', mineig/norm_v, self.t)
    
    def run(self, horizon: int, throttle: int=100, log_path: str=None) -> None:
        if log_path is None:
            log_path = f"tblogs/{type(self).__name__}_{self.env.dataset_name}"
        self.log_path = log_path
        self.writer = SummaryWriter(log_path)

        self._continue(horizon)
        postfix = {
            # 'total regret': 0.0,
            '% optimal arm (last 100 steps)': 0.0,
            'train loss': 0.0,
            'expected regret': 0.0
        }
        train_losses = []
        with tqdm(initial=self.t, total=horizon, postfix=postfix) as pbar:
            while (self.t < horizon):
                start = time.time()
                context = self.env.sample_context()
                features = self.env.features() #shape na x dim
                action = self.play_action(features=features)
                reward = self.env.step(action)
                #############################
                # add sample to replay buffer
                exp = (features[action], reward)
                self.buffer.append(exp)

                #############################
                # estimate linear component on the embedding + UCB part
                with torch.no_grad():
                    xt = torch.tensor(features[action].reshape(1,-1), dtype=TORCH_FLOAT).to(self.device)
                    v = self.target_model.embedding(xt).squeeze()
                    self.A += torch.outer(v.ravel(),v.ravel())
                    self.b_vec = self.b_vec + v * reward
                    self.inv_A, den = inv_sherman_morrison(v, self.inv_A)
                    # self.A_logdet += np.log(den)
                    self.theta = self.inv_A @ self.b_vec
                
                train_loss = 0
                if self.t > self.batch_size:
                    #############################
                    # train
                    self.model.train()
                    train_loss = []
                    for _ in range(self.max_updates):
                        features, rewards = self.buffer.sample(size=self.batch_size)
                        features = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
                        rewards = torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)
                        self.optimizer.zero_grad()
                        prediction = self.model(features)
                        mse_loss = F.mse_loss(prediction, rewards)
                        self.writer.add_scalar('mse_loss', mse_loss.item(), self.tot_update)
                        
                        phi = self.model.embedding(features)
                        # nv=torch.norm(phi,p=2,dim=1).max().cpu().detach().numpy()
                        A = torch.matmul(phi.transpose(1, 0), phi)
                        spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
                        self.writer.add_scalar('spectral_loss', spectral_loss, self.tot_update)

                        mse_weight = self.tot_update / (horizon/4) 
                        # mse_weight = (self.tot_update) / (self.tot_update + 10)
                        self.writer.add_scalar('mse_weight', mse_weight, self.tot_update)

                        loss = mse_weight * mse_loss - spectral_loss


                        loss.backward()
                        self.optimizer.step()
                        self.writer.flush()
                        self.tot_update += 1 
                        train_loss.append(loss.item())
                    train_loss = np.mean(train_loss)
                    self.model.eval()

                if self.t == self.update_time:
                    #############################
                    # copy to target
                    self.target_model.load_state_dict(self.model.state_dict())
                    self.target_model.eval()
                    # self.update_time = self.update_time + self.update_every 
                    self.update_time = int(np.ceil(max(1, self.update_time) * self.update_every))
                    self._update_after_change_of_target()
                
                self.runtime[self.t] = time.time() - start

                # log regret
                best_reward, best_action = self.env.best_reward_and_action()
                self.instant_reward[self.t] = reward 
                self.expected_reward[self.t] = self.env.expected_reward(action)
                self.best_reward[self.t] = best_reward

                rewards = [self.env.expected_reward(a) for a in range(self.env.action_space.n)]
                sorted = np.sort(rewards)
                self.action_gap[self.t] = sorted[-1]-sorted[-2]
                self.writer.add_scalar('action gap', self.action_gap[self.t], self.t)

                # log accuracy
                self.action_history[self.t] = action
                self.best_action_history[self.t] = best_action
                
                # log
                # postfix['total regret'] += self.best_reward[self.t] - self.instant_reward[self.t]
                postfix['expected regret'] += self.best_reward[self.t] - self.expected_reward[self.t]
                p_optimal_arm = np.mean(
                    self.action_history[max(0,self.t-100):self.t+1] == self.best_action_history[max(0,self.t-100):self.t+1]
                )
                postfix['% optimal arm (last 100 steps)'] = '{:.2%}'.format(p_optimal_arm)
 
                postfix['train loss'] = train_loss
                train_losses.append(train_loss)

                # self.writer.add_scalar("regret", postfix['total regret'], self.t)
                self.writer.add_scalar("expected regret", postfix['expected regret'], self.t)
                self.writer.add_scalar('perc optimal pulls (last 100 steps)', p_optimal_arm, self.t)
                self.writer.add_scalar('optimal arm?', 1 if self.action_history[self.t] == self.best_action_history[self.t] else 0, self.t)

                self.writer.flush()
                if self.t % throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(throttle)

                # step
                self.t += 1
        
        return {
            "regret": np.cumsum(self.best_reward - self.instant_reward),
            "optimal_arm": np.cumsum(self.action_history == self.best_action_history) / np.arange(1, len(self.action_history)+1),
            "expected_regret": np.cumsum(self.best_reward - self.expected_reward),
            "action_gap": self.action_gap,
            "runtime": self.runtime,
            "train_loss": train_losses

        }

    def _continue(self, horizon: int) -> None:
        """Continue learning from the point where we stopped
        """
        self.instant_reward = np.resize(self.instant_reward, horizon)
        self.expected_reward = np.resize(self.instant_reward, horizon)
        self.best_reward = np.resize(self.best_reward, horizon)
        self.action_history = np.resize(self.action_history, horizon)
        self.best_action_history = np.resize(self.best_action_history, horizon)
        self.action_gap = np.resize(self.action_gap, horizon)
        self.runtime = np.resize(self.runtime, horizon)