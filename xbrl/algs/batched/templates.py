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
import os
import pickle


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
        self.explorative_buffer = SimpleBuffer(capacity=self.buffer_capacity)
        # TODO: check the following lines: with initialization to 0 the training code is never called
        self.update_time = 2
        # self.update_time = 2**np.ceil(np.log2(self.batch_size)) + 1
        # self.update_time = int(self.batch_size + 1)
        self.number_glrt_step = 0


    def _post_train(self, loader=None) -> None:
        pass

    def add_sample(self, feature: np.ndarray, reward: float, all_features: np.ndarray) -> None:
        pass

    def compute_linear_error(self, features: np.ndarray, reward: np.ndarray):
        assert len(features.shape) == 2 and len(reward.shape) == 1
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
        rewards_tensor = torch.tensor(reward, dtype=TORCH_FLOAT).to(self.device)
        if self.model is not None:
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
        else:
            phi = features_tensor
        prediction = torch.matmul(phi, self.theta)
        error = prediction - rewards_tensor
        return error, phi
        # exp = (feature, reward, all_features.reshape( (1, self.env.action_space.n, self.env.feature_dim)))
        # self.buffer.append(exp)

    # def train(self) -> float:
    #     if self.model is None:
    #         return 0
    #     # if self.t % self.update_every == 0 and self.t > self.batch_size:
    #     if self.t == self.update_time:
    #         # self.update_time = max(1, self.update_time) * 2
    #         if self.update_every > 5:
    #             self.update_time += self.update_every
    #         else:
    #             self.update_time = int(np.ceil(max(1, self.update_time) * self.update_every))
    #
    #         if self.t > 10:
    #             if self.reset_model_at_train:
    #                 initialize_weights(self.model)
    #                 self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
    #                                                     weight_decay=self.weight_decay)
    #                 if self.unit_vector is not None:
    #                     self.unit_vector = torch.ones(self.model.embedding_dim, dtype=TORCH_FLOAT).to(self.device) / np.sqrt(
    #                         self.model.embedding_dim)
    #                     self.unit_vector.requires_grad = True
    #                     self.unit_vector_optimizer = torch.optim.SGD([self.unit_vector], lr=self.learning_rate)
    #
    #             self.model.train()
    #             epoch_metrics = defaultdict(list)
    #             # print(self.t+1, len(self.explorative_buffer))
    #             # print(f"{self.number_glrt_step}")
    #             max_updates = max(1, self.max_updates * int(len(self.buffer) / self.batch_size))
    #             for _ in range(max_updates):
    #                 exp_features, exp_rewards = self.explorative_buffer.sample(batch_size=self.batch_size) # change sample to guarantee that we always return batch_size data
    #                 exp_features_tensor = torch.tensor(exp_features, dtype=TORCH_FLOAT, device=self.device)
    #                 exp_rewards_tensor = torch.tensor(exp_rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)
    #                 features, rewards, all_features, steps, is_random_steps = self.buffer.sample(batch_size=self.batch_size)
    #                 features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
    #                 rewards_tensor = torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)
    #                 all_features_tensor = torch.tensor(all_features, dtype=TORCH_FLOAT, device=self.device)
    #                 weights_tensor = torch.ones((features.shape[0], 1)).to(self.device)
    #                 if self.train_reweight:
    #                     weights_tensor = torch.tensor(is_random_steps, dtype=TORCH_FLOAT, device=self.device)
    #                 metrics = self._train_loss(exp_features_tensor, exp_rewards_tensor, features_tensor, rewards_tensor, weights_tensor, all_features_tensor)
    #                 # update epoch metrics
    #                 for key, value in metrics.items():
    #                     epoch_metrics[key].append(value)
    #                 self.writer.flush()
    #
    #             self.model.eval()
    #             self._post_train()
    #             # log to tensorboard
    #             if self.use_tb:
    #                 for key, value in epoch_metrics.items():
    #                     self.writer.add_scalar('epoch_' + key, np.mean(value[-100:]), self.t)
    #             if self.use_wandb:
    #                 wandb.log({'epoch_' + key: np.mean(value) for key, value in epoch_metrics.items()}, step=self.t)
    #             avg_loss = np.mean(epoch_metrics['train_loss'])
    #             return avg_loss
    #     return None

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

                exp_features, exp_rewards = self.explorative_buffer.get_all()
                features, rewards, all_features, steps, is_random_steps = self.buffer.get_all()
                nelem = features.shape[0]
                idxs = np.random.choice(exp_features.shape[0], size=nelem)
                exp_features_tensor = torch.tensor(exp_features[idxs], dtype=TORCH_FLOAT, device=self.device)
                exp_rewards_tensor = torch.tensor(exp_rewards[idxs].reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)
                features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
                rewards_tensor = torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.device)
                all_features_tensor = torch.tensor(all_features, dtype=TORCH_FLOAT, device=self.device)
                weights_tensor = torch.ones((features.shape[0], 1)).to(self.device)
                if self.train_reweight:
                    weights_tensor = torch.tensor(is_random_steps, dtype=TORCH_FLOAT, device=self.device)
                    # print(f"reweighting: avg: {weights_tensor.mean().cpu().item()} - min/max: {weights_tensor.min().cpu().item(), weights_tensor.max().cpu().item()}")

                torch_dataset = torch.utils.data.TensorDataset(exp_features_tensor,exp_rewards_tensor,features_tensor, rewards_tensor, weights_tensor, all_features_tensor)
                loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=self.batch_size, shuffle=True)
                self.model.train()

                for epoch in range(self.max_updates):
                    epoch_metrics = defaultdict(list)

                    for batch_exp_feat, batch_exp_rew, batch_features, batch_rewards, batch_weights, batch_all_features in loader:
                        metrics = self._train_loss(batch_exp_feat, batch_exp_rew, batch_features, batch_rewards, batch_weights, batch_all_features)
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

                # debug metric
                aux_metrics = {}
                aux_metrics["train_loss"] = avg_loss

                if hasattr(self.env, 'feature_matrix'):

                    num_context, num_action, dim = self.env.feature_matrix.shape
                    all_features = self.env.feature_matrix.reshape(-1, dim)
                    all_rewards = self.env.rewards.reshape(-1)
                    error, phi = self.compute_linear_error(all_features, all_rewards)
                    aux_metrics["max_err"] = torch.abs(error).max().item()
                    aux_metrics["mean_abs_err"] = torch.abs(error).mean().item()

                    # IS HLS
                    rank_phi = torch.linalg.matrix_rank(phi, tol=1e-4, hermitian=False).item()  # this works when # contexts > embedding dim
                    rows, cols = np.where((self.env.rewards.max(axis=1).reshape(-1, 1) - self.env.rewards) < 1e-4)
                    opt_phi = phi.reshape((num_context, num_action, self.model.embedding_dim))[rows, cols, :]
                    opt_A = torch.matmul(opt_phi.T, opt_phi) / opt_phi.shape[0]
                    aux_metrics["hls_rank"] = torch.linalg.matrix_rank(opt_A, tol=1e-4, hermitian=True).item()
                    aux_metrics["hls_lambda"] = torch.linalg.eigvalsh(opt_A).min().item()
                    aux_metrics["rank_phi"] = rank_phi

                    #
                    norm_opt_phi = F.normalize(opt_phi, dim=1)
                    norm_opt_A = torch.matmul(norm_opt_phi.T, norm_opt_phi) / norm_opt_phi.shape[0]
                    norm_phi = F.normalize(phi, dim=1)
                    aux_metrics["min_feat"] = (torch.matmul(norm_phi, norm_opt_A) * norm_phi).sum(axis=1).min().item()
                    # log to tensorboard
                    if self.use_tb:
                        for key, value in aux_metrics.items():
                            self.writer.add_scalar( key, value, self.t)
                    if self.use_wandb:
                        wandb.log({key: value for key, value in aux_metrics.items()}, step=self.t)

                return aux_metrics
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
                aux_metrics = self.train()

                # update metrics
                if aux_metrics:
                    for key, value in aux_metrics.items():
                        metrics[key].append(value)

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
                if aux_metrics:
                    if "train_loss" in aux_metrics.keys():
                        postfix['train loss'] = aux_metrics["train_loss"]

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
                if self.t % 1000 == 0:
                    metrics["optimal_arm"] = np.cumsum(np.array(metrics["expected_reward"]) == np.array(metrics["best_reward"])) / np.arange(1, len(
                        metrics["best_reward"]) + 1)
                    metrics['regret'] = np.cumsum(np.array(metrics["best_reward"]) - np.array(metrics["instant_reward"]))
                    metrics["expected_regret"] = np.cumsum(np.array(metrics["best_reward"]) - np.array(metrics["expected_reward"]))
                    with open(os.path.join(log_path, "latest_result.pkl"), 'wb') as f:
                        pickle.dump(metrics, f)

        # convert metrics to numpy.array
        for key, value in metrics.items():
            metrics[key] = np.array(value)
        # compute extra metrics
        metrics["optimal_arm"] = np.cumsum(metrics["expected_reward"] == metrics["best_reward"]) / np.arange(1, len(
            metrics["best_reward"]) + 1)
        metrics['regret'] = np.cumsum(metrics["best_reward"] - metrics["instant_reward"])
        metrics["expected_regret"] = np.cumsum(metrics["best_reward"] - metrics["expected_reward"])
        return metrics
