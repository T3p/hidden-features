import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from ..replaybuffer import SimpleBuffer
from tqdm import tqdm

@dataclass
class XBModule(nn.Module):

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
        reset_model_at_train: Optional[bool]=True
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
        self.reset_model_at_train = reset_model_at_train

    def reset(self) -> None:
        self.t = 0
        self.buffer = SimpleBuffer(capacity=self.buffer_capacity)
        self.instant_reward = np.zeros(1)
        self.best_reward = np.zeros(1)
        self.action_history = np.zeros(1, dtype=int)
        self.best_action_history = np.zeros(1, dtype=int)
        self.batch_counter = 0
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        exp = (features, reward)
        self.buffer.append(exp)

    def train(self) -> float:

        if self.t > self.batch_size:
            cnt = 0
            sum_loss = 0
            for _ in range(self.max_updates):
                features, rewards = self.buffer.sample(self.batch_size)
                b_features = torch.FloatTensor(features, device=self.device)
                b_rewards = torch.FloatTensor(rewards.reshape(-1,1), device=self.device)
                loss = self._train_loss(b_features, b_rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.flush()
                self.batch_counter += 1
                cnt += 1
                sum_loss += loss.item()
            return sum_loss / cnt
        return None

    def _continue(self, horizon: int) -> None:
        """Continue learning from the point where we stopped
        """
        self.instant_reward = np.resize(self.instant_reward, horizon)
        self.best_reward = np.resize(self.best_reward, horizon)
        self.action_history = np.resize(self.action_history, horizon)
        self.best_action_history = np.resize(self.best_action_history, horizon)

    def run(self, horizon: int, throttle: int=100, log_path: str=None) -> None:
        if log_path is None:
            log_path = f"tblogs/{type(self).__name__}_{self.env.dataset_name}"
        self.log_path = log_path
        self.writer = SummaryWriter(log_path)

        self._continue(horizon)
        postfix = {
            'total regret': 0.0,
            '% optimal arm (last 100 steps)': 0.0,
            'train loss': 0.0
        }
        with tqdm(initial=self.t, total=horizon, postfix=postfix) as pbar:
            while (self.t < horizon):
                context = self.env.sample_context()
                features = self.env.features() #shape na x dim
                action = self.play_action(features=features)
                reward = self.env.step(action)
                # update
                self.add_sample(context, action, reward, features[action])
                train_loss = self.train()

                # log regret
                best_reward, best_action = self.env.best_reward_and_action()
                self.instant_reward[self.t] = reward #self.env.expected_reward(action)
                self.best_reward[self.t] = best_reward

                # log accuracy
                self.action_history[self.t] = action
                self.best_action_history[self.t] = best_action
                
                # log
                postfix['total regret'] += self.best_reward[self.t] - self.instant_reward[self.t]
                p_optimal_arm = np.mean(
                    self.action_history[max(0,self.t-100):self.t+1] == self.best_action_history[max(0,self.t-100):self.t+1]
                )
                postfix['% optimal arm (last 100 steps)'] = '{:.2%}'.format(p_optimal_arm)
                if train_loss:
                    postfix['train loss'] = train_loss

                self.writer.add_scalar("regret", postfix['total regret'], self.t)
                self.writer.add_scalar('perc optimal arm', p_optimal_arm, self.t)
                self.writer.add_scalar('optimal arm?', 1 if self.action_history[self.t] == self.best_action_history[self.t] else 0, self.t)
                self.writer.flush()

                if self.t % throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(throttle)

                # step
                self.t += 1
        
        return {
            "regret": np.cumsum(self.best_reward - self.instant_reward),
            "optimal_arm": np.cumsum(self.action_history == self.best_action_history) / np.arange(1, len(self.action_history)+1)
        }
