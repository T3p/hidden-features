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
from omegaconf import DictConfig

class NNLeaderInc(NNLinUCBInc):

    def __init__(self, env: Any,
                 model: nn.Module,
                 cfg: DictConfig
                 ):
        super().__init__(env, model, cfg)

    def _compute_loss(self, features, rewards):
        prediction = self.model(features)
        mse_loss = F.mse_loss(prediction, rewards)
        self.writer.add_scalar('mse_loss', mse_loss.item(), self.tot_update)

        phi = self.model.embedding(features)
        # nv=torch.norm(phi,p=2,dim=1).max().cpu().detach().numpy()
        A = torch.matmul(phi.transpose(1, 0), phi)
        spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
        self.writer.add_scalar('spectral_loss', spectral_loss, self.tot_update)

        mse_weight = self.tot_update / (self.horizon/2) 
        # mse_weight = (self.tot_update) / (self.tot_update + 10)
        self.writer.add_scalar('mse_weight', mse_weight, self.tot_update)

        loss = mse_weight * mse_loss - spectral_loss
        return loss
