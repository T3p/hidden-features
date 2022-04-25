import xbrl.envs as bandits
import matplotlib.pyplot as plt
from xbrl.algs.linear import LinUCB
from xbrl.algs.generalized_linear import UCBGLM, UCBGLM_general, OL2M
from xbrl.algs.batched.nnlinucb import NNLinUCB
from xbrl.algs.linear import LinUCB
from xbrl.envs.hlsutils import is_hls, derank_hls, normalize_linrep
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse
import json
import os
from scipy.special import logit
from scipy.special import expit as sigmoid
#from algs.nnmodel import Network

if __name__ == "__main__":
    seed = 0
    horizon = 500000
    rng = np.random.RandomState(seed=seed)
    
    #"""
    nc = 20
    na = 4
    dim = 5
    #features = np.load("problem_data/basic_features.npy")
    #dim = features.shape[-1]
    features = rng.uniform(low=-1., high=1., size=(nc, na, dim))
    #param = 0.01 * np.load("problem_data/basic_param.npy")
    param = rng.uniform(low=-1., high=1., size=dim)
    #"""

    rewards = features @ param
    assert is_hls(features, rewards)
    
    #"""
    env = bandits.CBFinite(feature_matrix=features,
                           rewards=rewards,
                           noise="bernoulli",
                           seed=seed)
    min_gap=env.min_suboptimality_gap()
    exploration_rounds = np.sqrt(dim*horizon)
    print(exploration_rounds)

    algo = UCBGLM(
        env=env,
        seed=seed,
        update_every_n_steps=1,
        delta=0.01,
        ucb_regularizer=1.,
        bonus_scale=1.,
        opt_tolerance=1e-8,
        true_param=None,
        param_bound = np.linalg.norm(param),
        exploration_rounds=exploration_rounds
    )
    algo.reset()
    result = algo.run(horizon=horizon)
    regrets = result['expected_regret']
    plt.plot(regrets)
    """ 
    env = bandits.CBFinite(feature_matrix=features,
                           rewards=rewards,
                           noise="gaussian",
                           seed=args.seed,
                           noise_param=1.)
    min_gap=env.min_suboptimality_gap()
    algo = LinUCB(
        env=env,
        seed=args.seed,
        update_every_n_steps=1,
        delta=0.01,
        ucb_regularizer=1.,
        bonus_scale=1.
    )
    algo.reset()
    result = algo.run(horizon=args.horizon)
    regrets = result['expected_regret']
    plt.plot(regrets)
    #"""
