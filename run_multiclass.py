import envs as bandits
from algs.nnlinucb import NNLinUCB
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    env = bandits.make("mushroom_num", bandit_model="expanded")
    T = len(env)
    net = Network(env.feature_dim, hidden_size=100)
    algo = NNLinUCB(
        env=env,
        model=net,
        batch_size=64,
        max_epochs=3,
        update_every_n_steps=100,
        learning_rate=0.1,
        buffer_capacity=T,
        noise_std=1,
        delta=0.01,
        weight_l2param=1.,
        weight_mse=1
    )
    algo.reset()
    results = algo.run(horizon=T)

    plt.figure()
    plt.plot(results['regret'])
    plt.title('Regret')
    plt.figure()
    plt.plot(results['optimal_arm'])
    plt.title('Optimal Arm')
    plt.show()

