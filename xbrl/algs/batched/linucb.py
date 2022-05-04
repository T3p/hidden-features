
from typing import Optional, Any
from .nnlinucb import NNLinUCB
from omegaconf import DictConfig


class LinUCB(NNLinUCB):
    def __init__(
            self,
            env: Any,
            cfg: DictConfig,
    ) -> None:
        super().__init__(env, cfg)

    def train(self) -> None:
        pass

