import pdb

import numpy as np
from typing import Optional, Any
import torch
import torch.nn as nn
from omegaconf import DictConfig
import wandb

from .nnlinucb import NNLinUCB, TORCH_FLOAT


class NNEpsGreedy(NNLinUCB):
    def __init__(
        self, env: Any,
            cfg: DictConfig,
            model: Optional[nn.Module] = None
    ) -> None:
        super().__init__(env, cfg, model)
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_decay = cfg.epsilon_decay
        self.time_random_exp = cfg.time_random_exp
        self.np_random = np.random.RandomState(self.seed)

        # initialization
        self.epsilon = self.epsilon_start


    def play_action(self, features: np.ndarray) -> int:
        # if self.t > self.time_random_exp and self.epsilon > self.epsilon_min:
            # self.epsilon -= (self.epsilon_start - self.epsilon_min) / self.epsilon_decay
        self.is_random_step = 0
        if self.epsilon_decay == "cbrt":
            self.epsilon = 1. / np.cbrt(self.t + 1)
        elif self.epsilon_decay == "sqrt":
            self.epsilon = 1. / np.sqrt(self.t + 1)
        elif self.epsilon_decay == "frt":
            self.epsilon = 1. / np.power(self.t + 1, 0.25)
        elif self.epsilon_decay == "zero":
            self.epsilon = 0
        else:
            raise NotImplementedError()
        
        glrt_active, min_ratio, beta, action = self.glrt(features)
        glrt_active = glrt_active and self.check_glrt and (len(self.explorative_buffer)>self.batch_size)

        if self.use_tb:
            self.writer.add_scalar('epsilon', self.epsilon, self.t)
            self.writer.add_scalar('GRLT', int(glrt_active), self.t)
        if self.use_wandb:
            wandb.log({'epsilon': self.epsilon}, step=self.t)
            wandb.log({'GRLT': int(glrt_active)}, step=self.t)
        if glrt_active:
            self.number_glrt_step += 1
            return action
        elif self.np_random.rand() <= self.epsilon:
            self.is_random_step = 1
            return self.np_random.choice(self.env.action_space.n, size=1).item()
        else:
            self.is_random_step = 1
            features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.device)
            if self.model is not None:
                with torch.no_grad():
                    phi = self.model.embedding(features_tensor)
            else:
                phi = features_tensor
            predicted_rewards = torch.matmul(phi, self.theta)
            opt_arms = torch.where(predicted_rewards > predicted_rewards.max() - self.mingap_clip)[0]
            action = self.np_random.choice(opt_arms.cpu().detach().numpy().flatten()).item()
            return action

