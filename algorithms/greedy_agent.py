from algorithms.models import MLP
from algs.replaybuffer import SimpleBuffer

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np


class GreedyAgent():
    def __init__(self, name, hparams):
        self.name = name
        self.hparams = hparams
        self.training_freq = hparams.training_freq
        self.training_steps = hparams.training_steps
        self.device = hparams.device
        self.t = 0

        self.replay_buffer = SimpleBuffer(capacity=hparams.buffer_size)
        self.model = MLP(hparams)
        self.optimizer = optim.Adam(self.model.parameters, lr=hparams.initial_lr)

    def action(self, context):
        """Selects action for context based the reward model prediction."""
        if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
            # round robin until each action has been taken "initial_pulls" times
            return self.t % self.hparams.num_actions

        context = Variable(torch.FloatTensor(context))
        context = context.to(self.device)
        output = self.model(context)
        return np.argmax(output.cpu().data)

    def update(self, context, action, reward):
        """Updates data buffer, and re-trains the neural model every training_freq steps."""
        self.t += 1
        self.replay_buffer.append((context[action], reward))
        if self.t % self.training_freq == 0:
            # print('{} pulls: training {} for {} steps ...'.format(self.t, self.name, self.training_steps))
            t_loss = []
            for _ in range(self.training_steps):
                x, y = self.replay_buffer.sample(self.hparams.batch_size)

                X = Variable(torch.FloatTensor(x))
                X = X.to(self.device)
                Y = Variable(torch.FloatTensor(y))
                Y = Y.to(self.device)
                Y_pred = self.model(X)
                Y_pred = Y_pred.squeeze()
                loss = torch.pow(Y_pred - Y, 2).mean()

                # print('true', Y.cpu().data)
                # print('pred', Y_pred.cpu().data)

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(self.network.parameters(), self.hparams.max_grad_norm)
                # for param in self.online_net.parameters():
                #     param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                t_loss.append(loss.item())
            return np.mean(t_loss)



