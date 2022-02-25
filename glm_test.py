import envs as bandits
import matplotlib.pyplot as plt
from algs.linear import LinUCB
from algs.generalized_linear import UCBGLM, OL2M
from algs.batched.nnlinucb import NNLinUCB
from algs.linear import LinUCB
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse
import json
import os
from algs.nnmodel import Network

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Bandit Test')
    # env options
    parser.add_argument('--dim', type=int, default=6, metavar='Context dimension')
    parser.add_argument('--narms', type=int, default=5, metavar='Number of actions')
    parser.add_argument('--horizon', type=int, default=100000, metavar='Horizon of the bandit problem')
    parser.add_argument('--seed', type=int, default=0, metavar='Seed used for the generation of the bandit problem')
    parser.add_argument('--bandittype', default="expanded", help="None, expanded, onehot")
    parser.add_argument('--contextgeneration', default="uniform", help="uniform, gaussian, bernoulli")
    # algo options
    parser.add_argument('--algo', type=str, default="ucbglm", help='algorithm [nnlinucb, nnleader]')
    parser.add_argument('--bonus-scale', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=100,
                        help="dimension of each layer (example --layers 100 200)")
    parser.add_argument('--max_epochs', type=int, default=10, help="maximum number of epochs")
    parser.add_argument('--update_every', type=int, default=100, help="Update every N samples")
    parser.add_argument('--config_name', type=str, default="", help='configuration name used to create the log')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")


    args = parser.parse_args()
    noise_std = 1.
    
    env = bandits.LinearContinuous(
        context_dim=args.dim, num_actions=args.narms, context_generation=args.contextgeneration,
        feature_expansion=None, seed=args.seed, noise="bernoulli", noise_param=noise_std
    )

    
    algo = OL2M(
        env=env,
        seed=args.seed,
        update_every_n_steps=1,
        delta=0.01,
        ucb_regularizer=1,
        bonus_scale=1.,
        #step_size=0.01
    )
    algo.reset()
    result = algo.run(horizon=args.horizon)
    regrets = result['expected_regret']
    plt.plot(regrets)
