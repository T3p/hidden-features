import envs as bandits
import matplotlib.pyplot as plt
from algs.linear import LinUCB
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiClass Bandit Test')
    parser.add_argument('--dim', type=int, default=20, metavar='Context dimension')
    parser.add_argument('--narms', type=int, default=5, metavar='Number of actions')
    parser.add_argument('--horizon', type=int, default=10000, metavar='Horizon of the bandit problem')
    parser.add_argument('--seed', type=int, default=0, metavar='Seed used for the generation of the bandit problem')
    parser.add_argument('--bandittype', default="onehot", help="None, expanded, onehot")
    parser.add_argument('--contextgeneration', default="uniform", help="uniform, gaussian, bernoulli")

    args = parser.parse_args()
    noise_std = 0.5
    env = bandits.LinearContinuous(
        context_dim=args.dim, num_actions=args.narms, context_generation=args.contextgeneration,
        feature_expansion=args.bandittype, seed=args.seed, noise="gaussian", noise_param=noise_std
    )


    algo = LinUCB(
        env=env,
        seed=args.seed,
        update_every_n_steps=1,
        noise_std=noise_std,
        delta=0.01,
        ucb_regularizer=1,
        bonus_scale=1.
    )
    algo.reset()
    algo.run(horizon=args.horizon)
    