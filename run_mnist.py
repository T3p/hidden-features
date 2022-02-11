import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from envs.multiclass import MulticlassToBandit
from algs.nnlinucb import NNLinUCB

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.emb_context = 128
        self.emb_action = 64
        self.fc1 = nn.Linear(9216, self.emb_context)
        self.fc2 = nn.Sequential(
            nn.Linear(10, self.emb_action),
            nn.ReLU()
        )
        self.embedding_dim = self.emb_context + self.emb_action
        self.fc3 = nn.Linear(self.embedding_dim, 1)

    def embedding(self, context, action):
        x = self.conv1(context)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        context_embedding = self.fc1(x)
        action_embedding = self.fc2(action)
        x = torch.cat((context_embedding, action_embedding), dim=1)
        return x

    def forward(self, context, action):
        x = self.embedding(context, action)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=len(dataset1))
    T = len(dataset1)
    X = next(iter(train_loader))[0].numpy()
    y = next(iter(train_loader))[1].numpy()
    del train_loader
    del dataset1
    print(X.shape, y.shape)

    print("loaded datase")
    env = MulticlassToBandit(X=X, y=y, dataset_name="MNIST", seed=0, noise=None)
    algo = NNLinUCB(
        env=env,
        net=Net(),
        batch_size=64,
        max_epochs=3,
        update_every_n_steps=500,
        learning_rate=0.1,
        buffer_capacity=T,
        use_onehotencoding=True,
        noise_std=1,
        delta=0.01,
        weight_l2param=1.,
        weight_mse=1
    )
    algo.reset()
    algo.run(horizon=int(T/10))

