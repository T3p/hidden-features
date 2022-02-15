import envs as bandits
from algs.nnlinucb import NNLinUCB
from algs.nnepsilongreedy import NNEpsGreedy
import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.nn.modules import Module
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Network(nn.Module):

    def __init__(self, input_size:int, layers_data:list):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.embedding_dim = layers_data[-1][0]
        self.fc2 = nn.Linear(self.embedding_dim, 1, bias=False)
    
    def embedding(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        return self.fc2(x)

def train_full(X, y, model, learning_rate=1e-2, weight_decay=0, max_epochs=10, batch_size=64, device="cpu"):
    writer = SummaryWriter(f"tblogs/train")
    torch_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float, device=device),
                torch.tensor(y.reshape(-1,1), dtype=torch.float, device=device)
                )

    loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    batch_counter = 0
    tot_loss = []
    for epoch in range(max_epochs):
        lh = []
        for b_features, b_rewards in loader:
            pred = model(b_features)
            loss = F.mse_loss(pred, b_rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_counter += 1
            lh.append(loss.item())
        writer.add_scalar("epoch_loss", np.mean(lh), epoch)
        if np.mean(lh) < 1e-3:
            break
        tot_loss.append(np.mean(lh))

    return {
        'loss': tot_loss
    }

if __name__ == "__main__":
    env = bandits.make_from_dataset("covertype", bandit_model="expanded")
    T = len(env)
    # T = 4000
    # env = bandits.Bandit_Linear(feature_dim=10, arms=5, noise=0.1, seed=0)
    net = Network(env.feature_dim, [(100, nn.ReLU())])
    print(net)

    X, Y = None, None
    for i in range(len(env)):
        x, y = env.__getitem__(i)
        if X is None:
            X = x 
            Y = y
        X = np.concatenate((X,x), axis=0)
        Y = np.concatenate((Y,y), axis=0)

    print(X.shape, Y.shape)

    # idx = np.random.randint(0, X.shape[0], size=8000)
    # idx = np.where(Y)
    # X = X[idx]
    # Y = Y[idx]
    # print(X.shape, Y.shape)

    results = train_full(
        X=X, y=Y, model=net, 
        learning_rate=1e-2, weight_decay=0.00001,
        max_epochs=1000, batch_size=64
    )
    plt.plot(results['loss'])
    plt.show()
    exit(0)


    # algo = NNLinUCB(
    #     env=env,
    #     model=net,
    #     batch_size=64,
    #     max_epochs=10,
    #     update_every_n_steps=100,
    #     learning_rate=0.01,
    #     buffer_capacity=T,
    #     noise_std=1,
    #     delta=0.01,
    #     weight_decay=1e-4,
    #     weight_mse=1,
    #     ucb_regularizer=1,
    #     bonus_scale=0.1
    # )
    # algo = NNEpsGreedy(
    #     env=env,
    #     model=net,
    #     batch_size=64,
    #     max_epochs=10,
    #     update_every_n_steps=100,
    #     learning_rate=0.01,
    #     buffer_capacity=T,
    #     epsilon_start=5,
    #     epsilon_min=0.05,
    #     epsilon_decay=2000,
    #     weight_decay=0
    # )
    algo.reset()
    results = algo.run(horizon=T)

    plt.figure()
    plt.plot(results['regret'])
    plt.title('Regret')
    plt.figure()
    plt.plot(results['optimal_arm'])
    plt.title('Optimal Arm')
    plt.show()

