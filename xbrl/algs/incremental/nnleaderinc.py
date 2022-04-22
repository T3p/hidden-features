import numpy as np
from typing import Optional, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from ..replaybuffer import SimpleBuffer
from ... import TORCH_FLOAT
from .nnlinucbinc import NNLinUCBInc

class NNLeaderInc(NNLinUCBInc):

    def __init__(self, env: Any, model: nn.Module, device: Optional[str] = "cpu", batch_size: Optional[int] = 256, max_updates: Optional[int] = 1, learning_rate: Optional[float] = 0.001, weight_decay: Optional[float] = 0, buffer_capacity: Optional[int] = 10000, seed: Optional[int] = 0, update_every: Optional[int] = 100, noise_std: float = 1, delta: Optional[float] = 0.01, ucb_regularizer: Optional[float] = 1, bonus_scale: Optional[float] = 1, use_tb: Optional[bool] = True, use_wandb: Optional[bool] = False) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, update_every, noise_std, delta, ucb_regularizer, bonus_scale, use_tb, use_wandb)

    def __init__(
        self,
        env: Any,
        model: nn.Module,
        device: Optional[str]="cpu",
        batch_size: Optional[int]=256,
        max_updates: Optional[int]=1,
        learning_rate: Optional[float]=0.001,
        weight_decay: Optional[float]=0,
        buffer_capacity: Optional[int]=10000,
        seed: Optional[int]=0,
        update_every: Optional[int] = 100,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.,
        use_tb: Optional[bool]=True,
        use_wandb: Optional[bool]=False
    ) -> None:
        super().__init__(env, model, device, batch_size, max_updates, learning_rate, weight_decay, buffer_capacity, seed, update_every, noise_std, delta, ucb_regularizer, bonus_scale, use_tb, use_wandb)

    
    def _compute_loss(self, features, rewards):
        prediction = self.model(features)
        mse_loss = F.mse_loss(prediction, rewards)
        self.writer.add_scalar('mse_loss', mse_loss.item(), self.tot_update)

        phi = self.model.embedding(features)
        # nv=torch.norm(phi,p=2,dim=1).max().cpu().detach().numpy()
        A = torch.matmul(phi.transpose(1, 0), phi)
        spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
        self.writer.add_scalar('spectral_loss', spectral_loss, self.tot_update)

        mse_weight = self.tot_update / (self.horizon/4) 
        # mse_weight = (self.tot_update) / (self.tot_update + 10)
        self.writer.add_scalar('mse_weight', mse_weight, self.tot_update)

        loss = mse_weight * mse_loss - spectral_loss
        return loss
