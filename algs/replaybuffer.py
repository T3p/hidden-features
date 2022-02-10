import numpy as np
from collections import deque, namedtuple

Experience = namedtuple(
    "Experience",
    field_names=["context", "action", "reward"],
)

class SimpleBuffer:

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.
        """
        self.buffer.append(experience)

    def get_all(self):
        indices = np.arange(len(self.buffer))
        contexts, actions, rewards = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.array(contexts),
            np.array(actions),
            np.array(rewards),
        )