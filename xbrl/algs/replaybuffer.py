import numpy as np
from collections import deque
from typing import Tuple


class SimpleBuffer:

    def __init__(self, capacity: int, seed: int) -> None:
        self.buffer = deque(maxlen=capacity)
        self.seed = seed
        self.np_random = np.random.RandomState(self.seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Tuple) -> None:
        """Add experience to the buffer.
        """
        self.buffer.append(experience)

    def get_all(self):
        return self.sample(batch_size=len(self.buffer), replace=False)

    def sample(self, batch_size:int, replace:bool=True):
        nelem = len(self.buffer)
        if batch_size > nelem:
            replace = True
        indices = self.np_random.choice(nelem, size=batch_size, replace=replace)
        out = (np.array(el) for el in zip(*(self.buffer[idx] for idx in indices)))
        return out
