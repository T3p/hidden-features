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
from ... import TORCH_FLOAT


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
        reset_model_at_train: Optional[bool]=True,
        update_every: Optional[int] = 100,
        train_reweight: Optional[bool]=False
    ) -> None:
        super().__init__()
        self.env = env
        self.model = model
        self.device = device
        self.batch_size = batch_size if batch_size is not None else 1
        self.max_updates = max_updates
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_capacity = buffer_capacity
        self.seed = seed
        self.reset_model_at_train = reset_model_at_train
        self.update_every = update_every
        self.unit_vector: Optional[torch.tensor] = None
        self.train_reweight = train_reweight

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
            self.model.to(TORCH_FLOAT)

        # TODO: check the following lines: with initialization to 0 the training code is never called
        self.update_time = 2
        # self.update_time = 2**np.ceil(np.log2(self.batch_size)) + 1
        # self.update_time = int(self.batch_size + 1)

    def _post_train(self, loader=None) -> None:
        pass

    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        exp = (features, reward)
        self.buffer.append(exp)

    def train(self) -> float:
        # if self.t % self.update_every == 0 and self.t > self.batch_size:
        if self.t == self.update_time:
            # self.update_time = max(1, self.update_time) * 2
            self.update_time = int(np.ceil(max(1, self.update_time) * self.update_every))
            if self.t > 10: #self.batch_size:
                # self.update_time = self.update_time + self.update_every 
                if self.reset_model_at_train:
                    initialize_weights(self.model)
                    if self.unit_vector is not None:
                        self.unit_vector = torch.ones(self.model.embedding_dim).to(self.device) / np.sqrt(
                            self.model.embedding_dim)
                        self.unit_vector.requires_grad = True
                        self.unit_vector_optimizer = torch.optim.SGD([self.unit_vector], lr=self.learning_rate)
                    # for layer in self.model.children():
                    #     if hasattr(layer, 'reset_parameters'):
                    #         layer.reset_parameters()
                features, rewards = self.buffer.get_all()

                weights = torch.ones((features.shape[0], 1)).to(self.device)
                if self.train_reweight:
                    with torch.no_grad():
                        xt = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
                        pred = self.model(xt).ravel()
                        z = (pred.cpu().detach().numpy() - rewards)**2
                        z = 1/(1 + np.exp(-z))
                        # z = z**3
                        weights = torch.tensor(z, dtype=TORCH_FLOAT, device=self.device)
                print(torch.mean(weights), torch.max(weights), torch.min(weights))

                torch_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(features, dtype=TORCH_FLOAT, device=self.device),
                    torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device),
                    weights
                    )

                loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=self.batch_size, shuffle=True)
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                self.model.train()
                last_loss = 0.0
                for epoch in range(self.max_updates):
                    lh = []
                    for b_features, b_rewards, b_weights in loader:
                        loss = self._train_loss(b_features, b_rewards, b_weights)
                        if isinstance(loss, int):
                            break
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        self.writer.flush()
                        self.batch_counter += 1
                        lh.append(loss.item())
                    last_loss = np.mean(lh)
                    if last_loss < 1e-3:
                        break
                self.writer.add_scalar('epoch_mse_loss', last_loss, self.t)
                self.model.eval()
                self._post_train(loader)
                return last_loss
        return None

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
                # update
                self.add_sample(context, action, reward, features[action])
                train_loss = self.train()
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
                if train_loss:
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
