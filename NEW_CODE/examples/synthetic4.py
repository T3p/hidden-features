#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:31:16 2022
Based on the contruction in app. B.2 from http://proceedings.mlr.press/v139/papini21a.html
"""
import numpy as np
from xb.envs.contextualfinite import ContextualFinite
from xb.envs.synthetic import LinearRandom
from xb.envs.synthetic.linutils import is_hls, make_hls_rank, hls_rank
from xb.algorithms import LinUCB, LinLEADER, LinLeaderSelect
from xb.runner import Runner
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

T = 10000
SEED = 0#97764652
np.random.seed(SEED)
std = 0.3
n_runs = 5
env_seeds = [np.random.randint(99999) for _ in range(n_runs)]
delta = 0.01
dim = 6

env = LinearRandom(n_contexts=20, 
                   n_actions=5, 
                   feature_dim=dim, 
                   random_state=SEED,
                   noise_std=std)
assert is_hls(env.features, env.param)
rewards = env.labels

reps = []
ranks = []
for i in range(1, env.feat_dim+1):
    new_features, new_param = make_hls_rank(rewards, 
                                            dim=env.feat_dim, 
                                            rank=i,
                                            transform=True, 
                                            normalize=True, 
                                            seed=SEED)
    rank = hls_rank(new_features, new_param)
    assert rank == i
    ranks.append(rank)
    assert new_features.shape[-1] == env.feat_dim
    reps.append(ContextualFinite._rep(new_features))
assert is_hls(new_features, new_param)

algs = [LinUCB(rep=r, 
               reg_val=1., 
               noise_std=std, 
               features_bound=r.feature_bound(), 
               param_bound= 1., 
               delta=delta, 
               adaptive_ci=True, 
               random_state=SEED,
) for r in reps]

algs.append(LinLEADER(reps=reps,
                      reg_val=1.,
                      noise_std=std,
                      features_bound=[r.feature_bound() for r in reps],
                      param_bound= 1.,
                      delta=delta,
                      adaptive_ci=True,
                      random_state=SEED,
))

algs.append(LinLeaderSelect(reps=reps,
                            reg_val=1.,
                            noise_std=std,
                            features_bound=[r.feature_bound() for r in reps], 
                            param_bound= 1., 
                            delta=delta,
                            adaptive_ci=True,
                            random_state=SEED,
))

for i, algo in enumerate(algs):
    regrets = np.zeros((n_runs, T))
    if i < len(reps):
        name = (type(algo).__name__ + ' hls_rank=' + str(ranks[i])
            + (' (HLS)' if ranks[i]==dim else ''))
    else:
        name = type(algo).__name__
    print(f"Running {name}...")

    for j in range(n_runs):    
        env_copy = deepcopy(env)
        env_copy.rng = np.random.RandomState(env_seeds[j])
        runner = Runner(env=env_copy, algo=algo, T_max=T)
        runner.reset()
        out = runner()
        regrets[j] = out['expected_regret']

    mean = np.mean(regrets, axis=0)
    plt.plot(mean, next(linecycler), label=name)
    if n_runs > 1:
        low = mean - np.std(regrets, axis=0) / np.sqrt(n_runs)
        high = mean + np.std(regrets, axis=0) / np.sqrt(n_runs)
        plt.fill_between(np.arange(len(mean)), low, high, alpha=0.3)

plt.legend()
plt.show()
