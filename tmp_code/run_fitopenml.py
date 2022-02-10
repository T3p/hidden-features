import numpy as np
from torchleader_discrete import Critic
import openml
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.utils import check_X_y
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pytorch_lightning import loggers as pl_loggers


class LitOpenML(LightningModule):

    def __init__(self, 
        dataset_id,
        model: nn.Module, 
        batch_size:int=64,
        learning_rate:float=2e-4,
        weight_mse: float=1, 
        weight_spectral:float=1, 
        weight_l2features:float=1,
        weight_l2param:float=1.,
        test_size=0.75
    ):

        super().__init__()

        # Set our init args as class attributes
        self.dataset_id = dataset_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_mse = weight_mse
        self.weight_spectral = weight_spectral
        self.weight_l2features = weight_l2features
        self.weight_l2param = weight_l2param
        self.test_size = test_size

    def forward(self, x, a):
        x = self.model(x, a)
        return x

    def training_step(self, batch, batch_idx):
        x, a, y = batch
        loss = 0
        # MSE LOSS
        if not np.isclose(self.weight_mse,0):
            prediction = self(x, a)
            mse_loss = F.mse_loss(prediction, y)
            self.log("mse_loss", mse_loss, prog_bar=True)
            loss = loss + self.weight_mse * mse_loss

        #DETERMINANT or LOG_MINEIG LOSS
        if not np.isclose(self.weight_spectral,0):
            phi = self.model.features(x, a)
            A = torch.sum(phi[...,None]*phi[:,None], axis=0)
            # det_loss = torch.logdet(A)
            spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
            self.log("spectral_loss", spectral_loss, prog_bar=True)
            loss = loss + self.weight_spectral * spectral_loss

        # FEATURES NORM LOSS
        if not np.isclose(self.weight_l2features,0):
            l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
            # l2 reg on parameters can be done in the optimizer
            # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
            self.log("l2feat_loss", l2feat_loss, prog_bar=True)
            loss = loss + self.weight_l2features * l2feat_loss

        # TOTAL LOSS
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, a, y = batch
        prediction = self(x, a)
        loss = F.mse_loss(prediction, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_l2param)
        return optimizer

    def setup(self, stage=None):
        
        dataset = openml.datasets.get_dataset(self.dataset_id)
        Xx, yy, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )
        X, y = check_X_y(X=Xx, y=yy, ensure_2d=True, multi_output=False)
        # re-index actions from 0 to n_classes
        y = (rankdata(y, "dense") - 1).astype(int)
        n_classes = np.unique(y).shape[0]

        (
            X_pre,
            X_test,
            y_pre,
            y_test
        ) = train_test_split(
            X, y, test_size=self.test_size, random_state=0
        )
    
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            new_X, new_A, new_Y = generate_bandit_dataset(X_pre, y_pre, standardize=True, n_classes=n_classes)
            self.train_data = TensorDataset(
                torch.tensor(new_X, dtype=torch.float), 
                torch.tensor(new_A, dtype=torch.float), 
                torch.tensor(new_Y.reshape(-1,1), dtype=torch.float)
            )
            _, X_val, _, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
            new_X, new_A, new_Y = generate_bandit_dataset(X_val, y_val, standardize=True, n_classes=n_classes)
            self.val_data = TensorDataset(
                torch.tensor(new_X, dtype=torch.float), 
                torch.tensor(new_A, dtype=torch.float), 
                torch.tensor(new_Y.reshape(-1,1), dtype=torch.float)
            )

            self.model = Critic(new_X.shape[1], n_classes)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            new_X, new_A, new_Y = generate_bandit_dataset(X_test, y_test, standardize=True, n_classes=n_classes)
            self.test_data = TensorDataset(
                torch.tensor(new_X, dtype=torch.float), 
                torch.tensor(new_A, dtype=torch.float), 
                torch.tensor(new_Y.reshape(-1,1), dtype=torch.float)
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

def generate_bandit_dataset(X, y, n_classes, standardize=False):
    if standardize:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    assert len(y) == n_samples
    
    new_X = np.zeros((n_samples*n_classes, n_features))
    new_A = np.zeros((n_samples*n_classes, n_classes))
    new_y = np.zeros(n_samples*n_classes)
    
    for i in range(n_samples):
        for j in range(n_classes):
            one_hot = np.zeros(n_classes)
            one_hot[j] = 1.
            new_X[i*n_classes + j] = X[i]
            new_A[i*n_classes + j] = one_hot
            new_y[i*n_classes + j] = 1. if y[i] == j else 0.
    return new_X, new_A, new_y


if __name__ == "__main__":
    id = 41
    test_size = 0.75
    learning_rate = 0.01
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    batch_size = 256 if AVAIL_GPUS else 64

    model = LitOpenML(
        dataset_id = id,
        model=Critic, 
        batch_size = batch_size,
        learning_rate = learning_rate,
        weight_mse=1, 
        weight_spectral=0, 
        weight_l2features=0,
        weight_l2param=0,
        test_size=test_size
    )
    early_stopping = EarlyStopping('val_loss')
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=10,
        progress_bar_refresh_rate=20,
        # callbacks=[early_stopping]
    )
    trainer.fit(model)
    test_res = trainer.test(model)
    print(test_res)
