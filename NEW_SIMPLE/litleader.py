import numpy as np

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

from linearenv import LinearRepresentation

#https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/mnist-hello-world.html
#https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html#:~:text=Stopping%20an%20epoch%20early,will%20stop%20your%20entire%20run.


class SimpleCircData:

    def __init__(self, capacity:int, dim:int) -> None:
        self.capacity = capacity
        self.X = np.zeros((capacity, dim), dtype=float)
        self.Y = np.zeros((capacity,1), dtype=float)
        self.current_capacity = 0
        self.current_index = 0
    
    def __len__(self):
        return self.current_capacity
    
    def add(self, x, y):
        self.X[self.current_index] = x
        self.Y[self.current_index] = y
        self.current_index += 1
        self.current_index = self.current_index % self.capacity
        self.current_capacity = min(self.current_capacity+1, self.capacity)

    def get_all(self):
        if self.current_capacity == 0:
            return None
        elif self.current_capacity >= self.capacity:
            return self.X, self.Y
        else:
            return self.X[0:self.current_index], self.Y[0:self.current_index]



class LitTorchTrain(LightningModule):

    def __init__(self, 
        dataset: TensorDataset, 
        model: nn.Module, 
        batch_size:int=64,
        learning_rate:float=2e-4,
        weight_mse: float=1, 
        weight_spectral:float=1, 
        weight_l2features:float=1,
        weight_l2param:float=1.
    ):

        super().__init__()

        # Set our init args as class attributes
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.model = model
        self.batch_size = batch_size
        self.weight_mse = weight_mse
        self.weight_spectral = weight_spectral
        self.weight_l2features = weight_l2features
        self.weight_l2param = weight_l2param

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # MSE LOSS
        prediction = self(x)
        mse_loss = F.mse_loss(prediction, y)
        self.log("mse_loss", mse_loss, prog_bar=True)

        #DETERMINANT or LOG_MINEIG LOSS
        phi = self.net.features(x)
        A = torch.sum(phi[...,None]*phi[:,None], axis=0)
        # det_loss = torch.logdet(A)
        spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
        self.log("spectral_loss", spectral_loss, prog_bar=True)

        # FEATURES NORM LOSS
        l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
        # l2 reg on parameters can be done in the optimizer
        # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
        self.log("l2feat_loss", l2feat_loss, prog_bar=True)

        # TOTAL LOSS
        loss = self.weight_mse * mse_loss + self.weight_spectral * spectral_loss + self.weight_l2features * l2feat_loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     prediction = self(x)
    #     loss = F.mse_loss(prediction, y)

    #     # Calling self.log will surface up scalars for you in TensorBoard
    #     self.log("val_loss", loss, prog_bar=True)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_l2param)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################
    # def setup(self, stage=None):

    #     # Assign train/val datasets for use in dataloaders
    #     if stage == "fit" or stage is None:
    #         N = self.dataset.shape[0]
    #         n_train = int(0.9*N)
    #         self.d_train, self.d_val = random_split(self.dataset, [n_train, N-n_train])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.d_val, batch_size=self.batch_size)

class LitLeader:

    def __init__(
        self, env, 
        net: nn.Module,
        representation: LinearRepresentation, 
        noise_std: float,
        features_bound: float,
        param_bound: float, 
        delta:float=0.01, random_state:int=0,
        device: str="cpu", batch_size:int=256, max_epochs:int=1,
        weight_l2param: float=1.,
        weight_mse: float=1, weight_spectral:float=1, weight_l2features:float=1,
        buffer_capacity:int=10000,
        update_every_n_steps: int = 100,
        bonus_scale: float=1.
    ) -> None:
        self.env = env
        self.net = net
        self.rep = representation
        self.noise_std = noise_std
        self.features_bound = features_bound
        self.param_bound=param_bound
        self.delta = delta
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_l2param = weight_l2param
        self.weight_mse = weight_mse
        self.weight_spectral = weight_spectral
        self.weight_l2features = weight_l2features
        self.buffer_capacity = buffer_capacity
        self.update_every_n_steps = update_every_n_steps
        self.bonus_scale = bonus_scale
    
    def reset(self):
        self.t = 1
        dim = self.rep.features_dim()
        self.buffer = SimpleCircData(capacity=self.buffer_capacity, dim=dim)
        self.inv_A = np.eye(dim, dtype=float) / self.weight_l2param
        self.instant_reward = np.zeros(1)
        self.best_reward = np.zeros(1)

    @torch.no_grad()
    def action(self, context: np.ndarray, available_actions: np.ndarray):
        dim = self.rep.features_dim()
        beta = self.noise_std * np.sqrt(dim * np.log((1+self.features_bound*self.features_bound*self.t/self.weight_l2param)/self.delta)) + self.param_bound * np.sqrt(self.weight_l2param)

        n_arms = available_actions.shape[0]
        ref_feats = np.zeros((n_arms, dim))
        for i, a in enumerate(available_actions):
            v = self.rep.get_features(context, a)
            ref_feats[i] = v
        ref_feats = torch.tensor(ref_feats, dtype=torch.float, device=self.device)

        if len(self.buffer) >= self.batch_size:
            prediction = self.net.predict(ref_feats).detach().numpy().ravel()
            net_features = self.net.features(ref_feats).detach().numpy()
            ucb = np.einsum('...i,...i->...', net_features @ self.inv_A, net_features)
            ucb = np.sqrt(ucb)
            ucb = prediction + self.bonus_scale * beta * ucb
            action = np.argmax(ucb)
            assert 0 <= action < n_arms, ucb
        else:
            action = np.random.choice(n_arms, 1).item()
        return action

    def update(self, context: np.ndarray, action: int, reward: float):
        v = self.rep.get_features(context, action)
        self.buffer.add(v, reward)

        if self.t % self.update_every_n_steps == 0 and self.t > self.batch_size:
            # update model
            X, Y = self.buffer.get_all()
            torch_dataset = torch.utils.data.TensorDataset(X, Y)
            model = LitTorchTrain(
                dataset=torch_dataset,
                model=self.net,
                learning_rate=self.learning_rate,
                coeff_mselossloss=self.weight_mse,
                reg_spectralloss=self.weight_spectral,
                reg_l2featloss=self.weight_l2features,
                weight_l2param=0
            )
            AVAIL_GPUS = min(1, torch.cuda.device_count())
            early_stopping = EarlyStopping('train_loss')
            trainer = Trainer(
                gpus=AVAIL_GPUS,
                max_epochs=self.max_epochs,
                progress_bar_refresh_rate=20,
                callbacks=[early_stopping]
            )
            trainer.fit(model)

            # compute design matrix
            with torch.no_grad():
                dim = self.rep.features_dim()
                X, _ = self.buffer.get_all()
                phi = self.net.features(X.to(self.device)).detach().numpy()
                A = np.sum(phi[...,None]*phi[:,None], axis=0)
                self.A = A + self.reg_val * np.eye(dim)
                self.inv_A = np.linalg.inv(A)


    def _continue(self, horizon: int) -> None:
        """Continue learning from the point where we stopped
        """
        self.instant_reward = np.resize(self.instant_reward, horizon)
        self.best_reward = np.resize(self.best_reward, horizon)
    
    def run(self, horizon: int) -> None:

        self._continue(horizon)
        
        while (self.t < horizon):
            context = self.env.sample_context()
            avail_actions = self.env.get_available_actions()
            action = self.action(context=context, available_actions=avail_actions)
            reward = self.env.step(action)

            # update
            self.update(context, action, reward)

            # regret computation
            self.instant_reward[self.t] = self.env.expected_reward(action)
            self.best_reward[self.t] = self.env.best_reward()

            self.t += 1
        
        return {"regret": np.cumsum(self.best_reward - self.instant_reward)}