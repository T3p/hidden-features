import math
import torch
import torch.nn as nn
from typing import Tuple, List


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


class MLLinearNetwork(nn.Module):

    def __init__(self, input_size: int, layers_data: List[Tuple]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        if layers_data:
            for size, activation in layers_data:
                self.layers.append(nn.Linear(input_size, size))
                input_size = size  # For the next layer
                if activation is not None:
                    assert isinstance(activation, nn.Module), \
                        "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                    self.layers.append(activation)
            self.embedding_dim = layers_data[-1][0]
        else:
            self.embedding_dim = input_size
            self.layers = None
        self.fc2 = nn.Linear(self.embedding_dim, 1, bias=False)
        initialize_weights(self)

    def embedding(self, x):
        if self.layers:
            for layer in self.layers:
                x = layer(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        return self.fc2(x)

class LinearNetwork(MLLinearNetwork):

    def __init__(self, input_size: int):
        super().__init__(input_size, None)

# class LinearNetwork(nn.Module):
#     def __init__(self, input_size: int):
#         super().__init__()
#         self.fc = nn.Linear(input_size, 1, bias=False)
#         self.embedding_dim = input_size
#         initialize_weights(self)

#     def embedding(self, x):
#         return x

#     def forward(self, x):
#         return self.fc(x)



class MLLogisticNetwork(nn.Module):

    def __init__(self, input_size: int, layers_data: List[Tuple]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        if layers_data:
            for size, activation in layers_data:
                self.layers.append(nn.Linear(input_size, size))
                input_size = size  # For the next layer
                if activation is not None:
                    assert isinstance(activation, nn.Module), \
                        "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                    self.layers.append(activation)
            self.embedding_dim = layers_data[-1][0]
        else:
            self.layers = None
            self.embedding_dim = input_size
        self.fc2 = nn.Linear(self.embedding_dim, 1, bias=False)
        initialize_weights(self)

    def embedding(self, x):
        if self.layers:
            for layer in self.layers:
                x = layer(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        return torch.sigmoid(self.fc2(x))

class LogisticNetwork(MLLogisticNetwork):

    def __init__(self, input_size: int):
        super().__init__(input_size, None)