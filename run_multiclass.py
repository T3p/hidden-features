import envs as bandits
# from algs.batched import NNLinUCB, NNEpsGreedy, NNLeader
from algs.incremental import NNLinUCB, NNEpsGreedy, NNLeader
import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.nn.modules import Module
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
import argparse
import json
import os

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

### MOVE TO HYDRA FOR MAIN SCRIPT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiClass Bandit Test')

    parser.add_argument('--horizon', default=None, type=int, help='Horizon. None (default) => dataset size')
    parser.add_argument('--dataset', default='magic', metavar='DATASET')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--noise', type=str, default=None, help='noise type [None, "bernoulli", "gaussian"]')
    parser.add_argument('--noise_param', type=str, default=0.3, help='noise type [None, "bernoulli", "gaussian"]')
    parser.add_argument('--bandittype', default='extended', metavar='DATASET', help="expanded or onehot")
    parser.add_argument('--layers', nargs='+', type=int, default=100, help="dimension of each layer (example --layers 100 200)")
    parser.add_argument('--algo', type=str, default="nnleader", help='algorithm [nnlinucb, nnleader]')
    parser.add_argument('--max_epochs', type=int, default=10, help="maximum number of epochs")
    parser.add_argument('--update_every', type=int, default=500, help="Update every N samples")
    parser.add_argument('--config_name', type=str, default="", help='configuration name used to create the log')

    args = parser.parse_args()
    env = bandits.make_from_dataset(
        args.dataset, bandit_model="expanded", 
        seed=args.seed, noise=args.noise, noise_param=args.noise_param)
    print(f"Samples: {env.X.shape}")
    print(f'Labels: {np.unique(env.y)}')

    T = args.horizon
    if T is None:
        T = len(env)
    # T = 4000
    # env = bandits.Bandit_Linear(feature_dim=10, arms=5, noise=0.1, seed=0)
    print('layers: ', args.layers)
    hid_dim = args.layers
    if not isinstance(args.layers, list):
        hid_dim = [args.layers]
    layers = [(el, nn.ReLU()) for el in hid_dim]
    net = Network(env.feature_dim, layers)
    print(net)

    print(f'Input features dim: {env.feature_dim}')

    weight_decay = 1e-4
    bonus_scale = 0.5
    if args.algo == "nnlinucb":
        algo = NNLinUCB(
            env=env,
            model=net,
            batch_size=256,
            max_updates=args.max_epochs,
            update_every_n_steps=args.update_every,
            learning_rate=0.001,
            buffer_capacity=T,
            noise_std=1,
            delta=0.01,
            weight_decay=weight_decay,
            weight_mse=1,
            ucb_regularizer=1,
            bonus_scale=bonus_scale
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
    elif args.algo == "nnleader":
        algo = NNLeader(
            env=env,
            model=net,
            batch_size=256,
            max_updates=args.max_epochs,
            update_every_n_steps=args.update_every,
            learning_rate=0.001,
            buffer_capacity=T,
            noise_std=1,
            delta=0.01,
            weight_decay=weight_decay,
            weight_mse=1,
            weight_spectral=-0.1,
            weight_l2features=0,
            ucb_regularizer=1,
            bonus_scale=bonus_scale
        )
    algo.reset()
    log_path = f"tblogs/{type(algo).__name__}_{args.dataset}_{args.bandittype}{args.config_name}"
    isExist = os.path.exists(log_path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(log_path)

    config = vars(args)
    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(config,f, indent=4, sort_keys=True)

    results = algo.run(horizon=T, log_path=log_path)

    plt.figure()
    plt.plot(results['regret'])
    plt.title('Regret')
    plt.figure()
    plt.plot(results['optimal_arm'])
    plt.title('Optimal Arm')
    plt.show()

