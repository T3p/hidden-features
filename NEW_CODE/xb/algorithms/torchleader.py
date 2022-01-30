import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from replaybuffer import ReplayBufferInMemStorage, ReplayBuffer


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()


    def forward(self, obs, action):
        return q1, q2

class TorchLeader:

    def __init__(self, device, lr, feature_dim, hidden_dim, buffer_capacity) -> None:
        self.device = device
        self.buffer_storage = ReplayBufferInMemStorage(capacity=buffer_capacity)

        # models
        self.critic = Critic(feature_dim, hidden_dim)

        # optimizers
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def action(self, context, available_actions):

        # compute features
        X = torch.FloatTensor(context['values'].reshape(1,-1)).to(self.device)
        values = self.model(X)
        
    def update(self, context, action, reward):
        self.buffer_storage.append(context, action, reward)

