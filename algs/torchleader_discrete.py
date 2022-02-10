import imp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import OneHotEncoder
from replaybuffer import SimpleBuffer, Experience




    

@dataclass
class TorchLeaderDiscrete(XBTorchDiscrete):

    noise_std: float=1
    features_bound: float=1
    param_bound: float=1
    delta: Optional[float]=0.01
    weight_mse: Optional[float]=1
    weight_spectral: Optional[float]=1
    weight_l2features: Optional[float]=1
    bonus_scale: Optional[float]=1.

    def _train_loss(self, b_context, b_actions, b_rewards):
        loss = 0
        # MSE LOSS
        if not np.isclose(self.weight_mse,0):
            prediction = self.net(b_context, b_actions)
            mse_loss = F.mse_loss(prediction, b_rewards)
            self.writer.add_scalar('mse_loss', mse_loss, self.batch_counter)
            loss = loss + self.weight_mse * mse_loss 

        #DETERMINANT or LOG_MINEIG LOSS
        if not np.isclose(self.weight_spectral,0):
            phi = self.net.features(b_context, b_actions)
            A = torch.sum(phi[...,None]*phi[:,None], axis=0)
            # det_loss = torch.logdet(A)
            spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
            self.writer.add_scalar('spectral_loss', spectral_loss, self.batch_counter)
            loss = loss + self.weight_spectral * spectral_loss

        # FEATURES NORM LOSS
        if not np.isclose(self.weight_l2features,0):
            l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
            # l2 reg on parameters can be done in the optimizer
            # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
            self.writer.add_scalar('l2feat_loss', l2feat_loss, self.batch_counter)
            loss = loss + self.weight_l2features * l2feat_loss

        # TOTAL LOSS
        self.writer.add_scalar('loss', loss, self.batch_counter)
        return loss
    
