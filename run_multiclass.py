import envs as bandits
from algs.nnlinucb import NNLinUCB
from algs.nnepsilongreedy import NNEpsGreedy
import torch
import torch.nn as nn 
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

class Network(nn.Module):

    def __init__(self, dim, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_size, 1)
        self.embedding_dim = hidden_size
    
    def embedding(self, x):
        return self.fc1(x)

    def forward(self, x):
        x = self.embedding(x)
        return self.fc2(x)

def train_full(X, y, model, learning_rate=1e-2, weight_decay=0, max_epochs=10, batch_size=64, device="cpu"):
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
        tot_loss.append(np.mean(lh))

    return {
        'loss': tot_loss
    }

if __name__ == "__main__":
    env = bandits.make_from_dataset("mushroom_num", bandit_model="expanded")
    T = len(env)
    # T = 4000
    # env = bandits.Bandit_Linear(feature_dim=10, arms=5, noise=0.1, seed=0)
    net = Network(env.feature_dim, hidden_size=100)

    # X, Y = None, None
    # for i in range(len(env)):
    #     x, y = env.__getitem__(i)
    #     if X is None:
    #         X = x 
    #         Y = y
    #     X = np.concatenate((X,x), axis=0)
    #     Y = np.concatenate((Y,y), axis=0)

    # print(X.shape, Y.shape)

    # idx = np.random.randint(0, X.shape[0], size=8000)
    # idx = np.where(Y)
    # X = X[idx]
    # Y = Y[idx]
    # print(X.shape, Y.shape)

    # results = train_full(
    #     X=X, y=Y, model=net, 
    #     learning_rate=0.25, weight_decay=0,
    #     max_epochs=20, batch_size=64
    # )
    # plt.plot(results['loss'])
    # plt.show()

    algo = NNLinUCB(
        env=env,
        model=net,
        batch_size=64,
        max_epochs=10,
        update_every_n_steps=100,
        learning_rate=0.01,
        buffer_capacity=T,
        noise_std=1,
        delta=0.01,
        weight_decay=1e-4,
        weight_mse=1,
        ucb_regularizer=1,
        bonus_scale=0.1
    )
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

