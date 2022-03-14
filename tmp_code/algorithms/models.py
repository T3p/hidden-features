import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class MLP(nn.Module):
    def __init__(self, hparams):
        """Creates a Model object.
            Args:
              hparams: Hyper-parameters.
        """
        nn.Module.__init__(self)
        self.hparams = hparams
        self.device = hparams.device

        self.embedding_net = nn.Sequential()
        input_size = hparams.context_dim * hparams.num_actions
        for (i, size) in enumerate(hparams.layer_sizes):
            self.embedding_net.add_module('linear{}'.format(i), nn.Linear(input_size, size))
            self.embedding_net.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            input_size = size
        self.embedding_dim = hparams.layer_sizes[-1]
        self.linear_layer = nn.Linear(hparams.layer_sizes[-1], 1, bias=True)
        self.embedding_net.to(self.device)
        self.linear_layer.to(self.device)

        initialize_weights(self)
        embedding_params = [param for param in self.embedding_net.parameters()]
        linear_params = [param for param in self.linear_layer.parameters()]
        self.parameters = embedding_params + linear_params

        # self.optimizer = optim.Adam(self.network.parameters(), lr=hparams.initial_lr)

    def embedding(self, input):
        output = self.embedding_net(input)
        return output

    def forward(self, input):
        input = self.embedding(input)
        output = self.linear_layer(input)
        return output


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels + m.in_channels
            m.weight.data.normal_(0, math.sqrt(4. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.in_features + m.out_features
            m.weight.data.normal_(0, math.sqrt(4. / n))
            if m.bias is not None:
                m.bias.data.zero_()



