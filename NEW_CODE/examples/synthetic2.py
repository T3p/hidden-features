#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:31:16 2022
Figure 7 from http://proceedings.mlr.press/v139/papini21a.html
'Varying dimension'
"""
import numpy as np
from xb.envs.contextualfinite import ContextualFinite
from xb.envs.synthetic import LinearRandom
from xb.envs.synthetic.linutils import is_hls, random_transform, derank_hls, hls_rank, reduce_dim
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

env = LinearRandom(n_contexts=20, 
                   n_actions=5, 
                   feature_dim=6, 
                   random_state=SEED,
                   noise_std=std)
assert is_hls(env.features, env.param)


hls_features, hls_param = random_transform(env.features, env.param, normalize=True, seed=SEED)

reps = []

for i in range(2, env.feat_dim + 1):
    reduced_features, reduced_param = reduce_dim(hls_features, hls_param, newdim=i, transform=False, normalize=False)
    new_features, new_param =  derank_hls(reduced_features, reduced_param, newrank=1, transform=True, normalize=True, seed=SEED)
    assert hls_rank(new_features, new_param) == 1
    assert new_features.shape[-1] == i
    reps.append(ContextualFinite._rep(new_features))

reps.append(ContextualFinite._rep(hls_features))

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
        name = (type(algo).__name__ + ' dim=' + str(reps[i].features_dim()) 
            + (' (HLS)' if i==len(reps) - 1 else ''))
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
