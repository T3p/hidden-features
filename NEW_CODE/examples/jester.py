#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:31:16 2022
Figure 1 from http://proceedings.mlr.press/v139/papini21a.html
"""
import numpy as np
from xb.envs.contextualfinite import ContextualFinite
from xb.envs.synthetic import LinearCB
from xb.envs.synthetic.linutils import is_hls, random_transform, derank_hls, hls_rank
from xb.algorithms import LinUCB, LinLEADER, LinLeaderSelect
from xb.runner import Runner
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

T = 100000
SEED = 0#97764652
np.random.seed(SEED)
n_runs = 5
env_seeds = [np.random.randint(99999) for _ in range(n_runs)]
std = 0.3
delta = 0.1

#Ground truth
rep = np.load('../data/jester/jester_post_d33_span33.npz')
features, param = rep['features'], rep['theta']

env = LinearCB(features, 
                   param, 
                   rewards=features @ param, 
                   random_state=SEED,
                   noise_std=std)

rep_files = ['../data/jester/jester_post_d33_span33.npz', 
             '../data/jester/jester_post_d26_span26.npz',
             '../data/jester/jester_post_d24_span24.npz',
             '../data/jester/jester_post_d23_span23.npz',
             '../data/jester/jester_post_d20_span20.npz', # <- non HLS
             '../data/jester/jester_post_d17_span17.npz',
             '../data/jester/jester_post_d16_span16.npz']
#all representations are non-redundant

reps = [ContextualFinite._rep(np.load(rf)['features']) for rf in rep_files]


dims = [np.load(rf)['features'].shape[-1] for rf in rep_files]
ranks = [hls_rank(np.load(rf)['features'], np.load(rf)['theta']) 
             for rf in rep_files]
#"""

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
                            normalize=False
))

algs.append(LinLeaderSelect(reps=reps,
                            reg_val=1.,
                            noise_std=std,
                            features_bound=[r.feature_bound() for r in reps], 
                            param_bound= 1., 
                            delta=delta,
                            adaptive_ci=True,
                            random_state=SEED,
                            normalize=True
))

for i, algo in enumerate(algs):
    regrets = np.zeros((n_runs, T))
    if i < len(reps):
        name = (type(algo).__name__ + ' dim=' + str(dims[i])
            + (' (HLS)' if ranks[i]==dims[i] else ''))
    else:
        name = type(algo).__name__ + (' (normalized)' if i==len(algs)-1 else '')
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

#plt.xscale('log')
plt.legend()
plt.show()
#"""