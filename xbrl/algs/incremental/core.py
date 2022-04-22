import numpy as np
from typing import Optional, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from ..replaybuffer import SimpleBuffer
from ..nnmodel import initialize_weights
import time
import copy
from ... import TORCH_FLOAT
import wandb


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + torch.dot(u.T, Au)
    A_inv -= torch.outer(Au, Au) / (den)
    return A_inv, den

class IncBase(nn.Module):

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
        use_tb: Optional[bool]=True,
        use_wandb: Optional[bool]=False
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
        self.use_tb = use_tb
        self.use_wandb = use_wandb

    def reset(self) -> None:
        self.t = 0
        self.buffer = SimpleBuffer(capacity=self.buffer_capacity)
        self.instant_reward = np.zeros(1)
        self.best_reward = np.zeros(1)
        self.action_gap = np.zeros(1)
        self.action_history = np.zeros(1, dtype=int)
        self.best_action_history = np.zeros(1, dtype=int)
        self.runtime = np.zeros(1)
        if self.model:
            self.model.to(self.device)

        # TODO: check the following lines: with initialization to 0 the training code is never called
        # self.update_time = 0
        self.update_time = 2

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.tot_update = 0
    
    def _compute_loss(self, features, rewards):
        prediction = self.model(features)
        mse_loss = F.mse_loss(prediction, rewards)
        if self.use_tb:
            self.writer.add_scalar('mse_loss', mse_loss.item(), self.tot_update)
        if self.use_wandb:
            wandb.log({'mse_loss': mse_loss.item()}, step=self.tot_update)
        return mse_loss

    @torch.no_grad()
    def play_action(self, features: np.ndarray):
        pass
    
    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        exp = (features, reward)
        self.buffer.append(exp)

    def _update_after_change_of_target():
        pass
    
    def run(self, horizon: int, throttle: int=100, log_path: str=None) -> None:
        if log_path is None:
            log_path = f"tblogs/{type(self).__name__}_{self.env.dataset_name}"
        self.log_path = log_path
        self.writer = SummaryWriter(log_path)

        self._continue(horizon)
        self.horizon = horizon
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
                self.add_sample(context, action, reward, features[action])
                
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
                        loss = self._compute_loss(features=features, rewards=rewards)
                        loss.backward()
                        self.optimizer.step()
                        self.writer.flush()
                        self.tot_update += 1 
                        train_loss.append(loss.item())
                    train_loss = np.mean(train_loss)
                    self.model.eval()

                if self.t == self.update_time:
                    #############################
                    # self.update_time = self.update_time + self.update_every 
                    self.update_time = int(np.ceil(max(1, self.update_time) * self.update_every))
                    if self.t > self.batch_size:
                        # copy to target
                        self.target_model.load_state_dict(self.model.state_dict())
                        self.target_model.eval()
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

                if self.use_tb:
                    self.writer.add_scalar('action gap', self.action_gap[self.t], self.t)
                if self.use_wandb:
                    wandb.log({'action_gap': self.action_gap[self.t]}, step=self.t)

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

                if self.use_tb:
                    self.writer.add_scalar("expected regret", postfix['expected regret'], self.t)
                    self.writer.add_scalar('perc optimal pulls (last 100 steps)', p_optimal_arm, self.t)
                    self.writer.add_scalar('optimal arm?', 1 if self.action_history[self.t] == self.best_action_history[self.t] else 0, self.t)
                if self.use_wandb:
                    wandb.log({
                        "expected regret": postfix['expected regret'], 
                        'perc optimal pulls (last 100 steps)': p_optimal_arm,
                        'optimal arm?': 1 if self.action_history[self.t] == self.best_action_history[self.t] else 0}, step=self.t
                    )


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
