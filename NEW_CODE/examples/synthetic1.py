#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:31:16 2022
Figure 1 from http://proceedings.mlr.press/v139/papini21a.html
"""
import numpy as np
from xb.envs.contextualfinite import ContextualFinite
from xb.envs.synthetic import LinearRandom
from xb.envs.synthetic.linutils import is_hls, random_transform, derank_hls, hls_rank
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
n_runs = 20
env_seeds = [np.random.randint(99999) for _ in range(n_runs)]
std = 0.3

env = LinearRandom(n_contexts=20, 
                   n_actions=5, 
                   feature_dim=6, 
                   random_state=SEED,
                   noise_std=std)
assert is_hls(env.features, env.param)

hls_features, hls_param = random_transform(env.features, env.param, normalize=True)

reps = []

for i in range(1, env.feat_dim):
    new_features, new_param =  derank_hls(hls_features, hls_param, newrank=i, normalize=True)
    assert hls_rank(new_features, new_param) == i
    reps.append(ContextualFinite._rep(new_features))

reps.append(ContextualFinite._rep(hls_features))

delta = 0.1

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
        low = mean - 2 * np.std(regrets, axis=0) / n_runs
        high = mean + 2 * np.std(regrets, axis=0) / n_runs
        plt.fill_between(np.arange(len(mean)), low, high, alpha=0.3)

plt.legend()
plt.show()
