import numpy as np
from typing import Optional, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from .replaybuffer import SimpleBuffer
import time
import copy


class NNEGInc(nn.Module):

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
            epsilon_min: float=0.05,
            epsilon_start: float=2,
            epsilon_decay: float=200,
            time_random_exp: int=0
    ) -> None:
        super().__init__()
        self.env = env
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_updates = max_updates
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_capacity = buffer_capacity
        self.seed = seed
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.time_random_exp = time_random_exp
        self.np_random = np.random.RandomState(self.seed)

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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.tot_update = 0
        self.epsilon = self.epsilon_start
    
    @torch.no_grad()
    def play_action(self, features: np.ndarray) -> int:
        if self.t > self.time_random_exp and self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_start - self.epsilon_min) / self.epsilon_decay
        self.writer.add_scalar('epsilon', self.epsilon, self.t)
        if self.np_random.rand() < self.epsilon:
            action = self.np_random.choice(self.env.action_space.n, size=1).item()
        else:
            xt = torch.FloatTensor(features).to(self.device)
            scores = self.model(xt)
            action = torch.argmax(scores).item()
        assert 0 <= action < self.env.action_space.n
        return action
    
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
                
                train_loss = 0
                if self.t > self.batch_size:
                    #############################
                    # train
                    self.model.train()
                    train_loss = []
                    for _ in range(self.max_updates):
                        features, rewards = self.buffer.sample(size=self.batch_size)
                        features = torch.tensor(features, dtype=torch.float, device=self.device)
                        rewards = torch.tensor(rewards.reshape(-1, 1), dtype=torch.float, device=self.device)
                        self.optimizer.zero_grad()
                        prediction = self.model(features)
                        mse_loss = F.mse_loss(prediction, rewards)
                        mse_loss.backward()
                        self.optimizer.step()
                        self.writer.add_scalar('mse_loss', mse_loss.item(), self.tot_update)
                        self.writer.flush()
                        self.tot_update += 1 
                        train_loss.append(mse_loss.item())
                    train_loss = np.mean(train_loss)
                    self.model.eval()
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