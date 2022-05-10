import pdb
import wandb
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
from collections import defaultdict
import logging
from omegaconf import DictConfig


class XBModule():

    def __init__(
        self,
        env: Any,
        cfg: DictConfig,
        model: Optional[nn.Module] = None
    ) -> None:
        self.env = env
        self.model = model
        self.device = cfg.device
        self.batch_size = cfg.batch_size if cfg.batch_size is not None else 1
        self.max_updates = cfg.max_updates
        self.learning_rate = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.buffer_capacity = cfg.buffer_capacity
        self.seed = cfg.seed
        self.reset_model_at_train = cfg.reset_model_at_train
        self.update_every = cfg.update_every
        self.unit_vector: Optional[torch.tensor] = None
        self.train_reweight = cfg.train_reweight
        self.logger = logging.getLogger(__name__)
        self.use_tb = cfg.use_tb
        self.use_wandb = cfg.use_wandb
        self.glrt_scale = cfg.glrt_scale
        self.mingap_clip = cfg.mingap_clip
        if model is not None:
            self.model.to(self.device)
            self.model.to(TORCH_FLOAT)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                             weight_decay=self.weight_decay)
        self.t = 0
        self.buffer = SimpleBuffer(capacity=self.buffer_capacity)
        # TODO: check the following lines: with initialization to 0 the training code is never called
        self.update_time = 2
        # self.update_time = 2**np.ceil(np.log2(self.batch_size)) + 1
        # self.update_time = int(self.batch_size + 1)


    def _post_train(self, loader=None) -> None:
        pass

    def add_sample(self, feature: np.ndarray, reward: float, all_features: np.ndarray) -> None:
        pass
        # exp = (feature, reward, all_features.reshape( (1, self.env.action_space.n, self.env.feature_dim)))
        # self.buffer.append(exp)

    def train(self) -> float:
        if self.model is None:
            return 0
        # if self.t % self.update_every == 0 and self.t > self.batch_size:
        if self.t == self.update_time:
            # self.update_time = max(1, self.update_time) * 2
            if self.update_every > 5:
                self.update_time += self.update_every
            else:
                self.update_time = int(np.ceil(max(1, self.update_time) * self.update_every))
            if self.t > 10: #self.batch_size:
                # self.update_time = self.update_time + self.update_every 
                if self.reset_model_at_train:
                    initialize_weights(self.model)
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                                     weight_decay=self.weight_decay)
                    if self.unit_vector is not None:
                        self.unit_vector = torch.ones(self.model.embedding_dim, dtype=TORCH_FLOAT).to(self.device) / np.sqrt(
                            self.model.embedding_dim)
                        self.unit_vector.requires_grad = True
                        self.unit_vector_optimizer = torch.optim.SGD([self.unit_vector], lr=self.learning_rate)

                features, rewards, all_features, steps, is_random_steps = self.buffer.get_all()
                features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
                rewards_tensor = torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)
                all_features_tensor = torch.tensor(all_features, dtype=TORCH_FLOAT, device=self.device)
                weights_tensor = torch.ones((features.shape[0], 1)).to(self.device)
                if self.train_reweight:
                    weights_tensor = torch.tensor(is_random_steps, dtype=TORCH_FLOAT, device=self.device)
                    print(f"reweighting: avg: {weights_tensor.mean().cpu().item()} - min/max: {weights_tensor.min().cpu().item(), weights_tensor.max().cpu().item()}")

                torch_dataset = torch.utils.data.TensorDataset(features_tensor, rewards_tensor, weights_tensor, all_features_tensor)
                loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=self.batch_size, shuffle=True)
                self.model.train()

                for epoch in range(self.max_updates):
                    epoch_metrics = defaultdict(list)

                    for batch_features, batch_rewards, batch_weights, batch_all_features in loader:
                        metrics = self._train_loss(batch_features, batch_rewards, batch_weights, batch_all_features)
                        # update epoch metrics
                        for key, value in metrics.items():
                            epoch_metrics[key].append(value)
                        self.writer.flush()

                self.model.eval()
                self._post_train(loader)
                # log to tensorboard
                if self.use_tb:
                    for key, value in epoch_metrics.items():
                        self.writer.add_scalar('epoch_' + key, np.mean(value), self.t)
                if self.use_wandb:
                    wandb.log({'epoch_' + key: np.mean(value) for key, value in epoch_metrics.items()}, step=self.t)
                avg_loss = np.mean(epoch_metrics['train_loss'])
                return avg_loss
        return None

    def run(self, horizon: int, throttle: int=100, log_path: str=None) -> None:
        metrics = defaultdict(list)
        if log_path is None:
            log_path = f"tblogs/{type(self).__name__}_{self.env.dataset_name}"
        self.log_path = log_path
        self.writer = SummaryWriter(log_path)

        postfix = {
            # 'total regret': 0.0,
            '% optimal arm (last 100 steps)': 0.0,
            'train loss': 0.0,
            'expected regret': 0.0
        }
        with tqdm(initial=self.t, total=horizon, postfix=postfix) as pbar:
            while self.t < horizon:
                start = time.time()
                features = self.env.features() #shape na x dim
                action = self.play_action(features=features)
                reward = self.env.step(action)
                # update
                self.add_sample(features[action], reward, features)
                train_loss = self.train()

                # update metrics
                metrics['runtime'].append(time.time() - start)
                # log regret
                metrics['instant_reward'].append(reward)
                expected_reward = self.env.expected_reward(action)
                metrics['expected_reward'].append(expected_reward)
                metrics['action'].append(action)
                best_reward, best_action = self.env.best_reward_and_action()
                metrics['best_reward'].append(best_reward)
                metrics['best_action'].append(best_action)
                # metrics['instant_regret'].append(best_reward - reward)
                # metrics["instant_expected_regret"].append(best_reward - expected_reward)

                # update postfix
                postfix['expected regret'] += best_reward - expected_reward
                p_optimal_arm = np.mean(
                    np.abs(np.array(metrics['expected_reward'][max(0,self.t-100):self.t+1]) - np.array(metrics['best_reward'][max(0,self.t-100):self.t+1]) ) < 1e-6
                )
                postfix['% optimal arm (last 100 steps)'] = '{:.2%}'.format(p_optimal_arm)
                if train_loss:
                    postfix['train loss'] = train_loss
                    metrics['train_loss'].append(train_loss)

                # log to tensorboard
                # self.writer.add_scalar("regret", postfix['total regret'], self.t)
                if self.use_tb:
                    self.writer.add_scalar("expected regret", postfix['expected regret'], self.t)
                    self.writer.add_scalar('perc optimal pulls (last 100 steps)', p_optimal_arm, self.t)
                    self.writer.add_scalar('optimal arm?', 1 if action == best_action else 0, self.t)
                if self.use_wandb:
                    wandb.log({'expected regret': postfix['expected regret'],
                               'perc optimal pulls (last 100 steps)': p_optimal_arm,
                               'optimal arm?': int(expected_reward == best_reward)}, step=self.t)

                if self.t % throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(throttle)

                # step
                self.t += 1
        # convert metrics to numpy.array
        for key, value in metrics.items():
            metrics[key] = np.array(value)
        # compute extra metrics
        metrics["optimal_arm"] = np.cumsum(metrics["expected_reward"] == metrics["best_reward"]) / np.arange(1, len(
            metrics["best_reward"]) + 1)
        metrics['regret'] = np.cumsum(metrics["best_reward"] - metrics["instant_reward"])
        metrics["expected_regret"] = np.cumsum(metrics["best_reward"] - metrics["expected_reward"])
        return metrics
