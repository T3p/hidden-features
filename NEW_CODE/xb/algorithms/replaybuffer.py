import datetime
import io
import random
import traceback
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

class ReplayBufferInMemStorage:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, context, action, reward):
        self.buffer.append((context, action, reward))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        contexts, actions, rewards = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(contexts), np.array(actions), np.array(rewards))
    
# TODO: we can move to store data instead of keeping them in memory
# see https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

class ReplayBuffer(IterableDataset):

    def __init__(self, buffer):
        self.buffer = buffer

    def _sample(self):
        idx = np.random.choice(len(self.buffer), 1, replace=False)
        contexts, actions, rewards = self.buffer.buffer[idx]
        return (contexts, actions, rewards)

    def __iter__(self):
        while True:
            yield self._sample()
