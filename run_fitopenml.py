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
        dataset,
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
    #     logits = self(x)
    #     loss = F.mse_loss(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     self.accuracy(preds, y)

    #     # Calling self.log will surface up scalars for you in TensorBoard
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.log("val_acc", self.accuracy, prog_bar=True)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     # Here we just reuse the validation_step for testing
    #     return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_l2param)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

def generate_bandit_dataset(X,y, standardize=False):
    if standardize:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    assert len(y) == n_samples
    classes = y.values.unique()
    n_classes = len(classes)
    
    new_X = np.zeros((n_samples*n_classes, n_features+n_classes))
    new_y = np.zeros(n_samples*n_classes)
    
    for i in range(n_samples):
        for j in range(n_classes):
            one_hot = np.zeros(n_classes)
            one_hot[j] = 1.
            new_X[i*n_classes + j] = np.concatenate((X[i], one_hot))
            new_y[i*n_classes + j] = 1. if y[i] == classes[j] else 0.
    return new_X, new_y


if __name__ == "__main__":
    ID = 1  
    TEST_SIZE = 0.75
    SEED = 0

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    BATCH_SIZE = 256 if AVAIL_GPUS else 64

    dataset = openml.datasets.get_dataset(ID)
    Xx, yy, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    X, y = check_X_y(X=Xx, y=yy, ensure_2d=True, multi_output=False)
    # re-index actions from 0 to n_classes
    y = (rankdata(y, "dense") - 1).astype(int)

    (
        X_pre,
        X_test,
        y_pre,
        y_test
    ) = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    new_X, new_Y = generate_bandit_dataset(X, y, standardize=True)
    dataset = TensorDataset(
        torch.tensor(new_X, dtype=float),
        torch.tensor(new_Y, dtype=float)
    )
    early_stopping = EarlyStopping('train_loss')

    model = LitOpenML()
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=30,
        progress_bar_refresh_rate=20,
        callbacks=[early_stopping],
        logger=tb_logger
    )
    trainer.fit(model)