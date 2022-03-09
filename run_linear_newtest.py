import envs as bandits
import envs.hlsutils as hlsutils
import matplotlib.pyplot as plt
from algs.linear import LinUCB
from algs.batched.nnlinucb import NNLinUCB
from algs.batched.nnleader import NNLeader
from algs.nnmodel import LinearNetwork
from algs.batched.nnepsilongreedy import NNEpsGreedy
from algs.linear import LinUCB
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse
import json
import os
import pickle
from algs.nnmodel import MLLinearNetwork, MLLogisticNetwork
import torch
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Bandit Test')
    # env options
    parser.add_argument('--dim', type=int, default=20, metavar='Context dimension')
    parser.add_argument('--narms', type=int, default=5, metavar='Number of actions')
    parser.add_argument('--horizon', type=int, default=20000, metavar='Horizon of the bandit problem')
    parser.add_argument('--seed', type=int, default=0, metavar='Seed for noise')
    parser.add_argument('--seed-problem', type=int, default=0, metavar='Seed used for the generation of the bandit problem')
    parser.add_argument('--bandittype', default="expanded", help="None, expanded, onehot")
    parser.add_argument('--contextgeneration', default="uniform", help="uniform, gaussian, bernoulli")
    # algo options
    parser.add_argument('--algo', type=str, default="linucb", help='algorithm [linucb, nnlinucb, nnleader]')
    parser.add_argument('--bonus_scale', type=float, default=0.1)
    parser.add_argument('--layers', nargs='+', type=int, default=100,
                        help="dimension of each layer (example --layers 100 200)")
    parser.add_argument('--weight_spectral', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=20, help="maximum number of epochs")
    parser.add_argument('--update_every', type=int, default=100, help="Update every N samples")
    parser.add_argument('--config_name', type=str, default="", help='configuration name used to create the log')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay, ie L2 regularization of the parameters")
    parser.add_argument('--noise_std', type=float, default=0.2, help="standard deviation of the noise")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)


    args = parser.parse_args()
    env = bandits.LinearContinuous(
        context_dim=args.dim, num_actions=args.narms, context_generation=args.contextgeneration,
        feature_expansion=args.bandittype, seed=args.seed, noise="gaussian", noise_param=args.noise_std,
        seed_problem=args.seed_problem
    )

    features, theta = bandits.make_synthetic_features(
        n_contexts=1000, n_actions=args.narms, dim=args.dim,
        context_generation=args.contextgeneration, feature_expansion=args.bandittype,
        seed=args.seed_problem
    )
    rewards = features @ theta
    print(f"Original rep -> HLS rank: {hlsutils.hls_rank(features, rewards)} / {features.shape[2]}")
    print(f"Original rep -> is HLS: {hlsutils.is_hls(features, rewards)}")
    print(f"Original rep -> is CMB: {hlsutils.is_cmb(features, rewards)}")
    # features, theta = hlsutils.derank_hls(features=features, param=theta, newrank=6)
    # rewards = features @ theta
    # print(f"New rep -> HLS rank: {hlsutils.hls_rank(features, rewards)} / {features.shape[2]}")
    # print(f"New rep -> is HLS: {hlsutils.is_hls(features, rewards)}")
    # print(f"New rep -> is CMB: {hlsutils.is_cmb(features, rewards)}")

    env = bandits.CBFinite(
        feature_matrix=features, 
        rewards=rewards, seed=args.seed, 
        noise="gaussian", noise_param=args.noise_std
    )

    # set_seed_everywhere
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    print('layers: ', args.layers)
    hid_dim = args.layers
    if not isinstance(args.layers, list):
        hid_dim = [args.layers]
    layers = [(el, nn.Tanh()) for el in hid_dim]
    net = MLLinearNetwork(env.feature_dim, layers).to(args.device)
    # net = Network(env.feature_dim, layers).to(args.device)

    print(net)

    print(f'Input features dim: {env.feature_dim}')
    if args.algo == "nnlinucb":
        algo = NNLinUCB(
            env=env,
            model=net,
            device=args.device,
            batch_size=args.batch_size,
            max_updates=args.max_epochs,
            update_every_n_steps=args.update_every,
            learning_rate=args.lr,
            buffer_capacity=args.horizon,
            noise_std=1,
            delta=0.01,
            weight_decay=args.weight_decay,
            ucb_regularizer=1,
            bonus_scale=args.bonus_scale,
            reset_model_at_train=True
        )
    elif args.algo == "nnepsilon":
        algo = NNEpsGreedy(
            env=env,
            model=net,
            batch_size=args.batch_size,
            max_updates=args.max_epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            buffer_capacity=args.horizon,
            seed=args.seed,
            reset_model_at_train=True,
            update_every_n_steps=args.update_every,
            epsilon_min=0.05,
            epsilon_start=1,
            epsilon_decay=200
        )
    elif args.algo == "linucb":
        algo = LinUCB(
            env=env,
            seed=args.seed,
            update_every_n_steps=args.update_every,
            noise_std=args.noise_std,
            delta=0.01,
            ucb_regularizer=1,
            bonus_scale=args.bonus_scale
        )
    elif args.algo == "nnleader":
        algo = NNLeader(
            env=env,
            model=net,
            device=args.device,
            batch_size=args.batch_size,
            max_updates=args.max_epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            buffer_capacity=args.horizon,
            seed=args.seed,
            reset_model_at_train=True,
            update_every_n_steps=args.update_every,
            noise_std=args.noise_std,
            delta=0.01,
            ucb_regularizer=1,
            weight_mse=1,
            bonus_scale=args.bonus_scale,
            weight_spectral=args.weight_spectral,
            weight_l2features=0
        )

    algo.reset()
    result = algo.run(horizon=args.horizon, log_path=args.log_dir)
    regrets = result['expected_regret']
    plt.plot(regrets)
    plt.savefig(os.path.join(algo.log_path, 'regret.png'))

    if args.save_dir is not None:
        with open(os.path.join(args.save_dir, "arguments.pkl"), 'wb') as f:
            pickle.dump(args, f)
        with open(os.path.join(args.save_dir, "result.pkl"), 'wb') as f:
            pickle.dump(result, f)


