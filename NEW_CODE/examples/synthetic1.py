#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:31:16 2022
"""
import numpy as np
from xb.envs.synthetic import LinearRandom
from xb.envs.synthetic.linutils import is_hls

T = 10000
SEED = 97764652
np.random.seed(SEED)

env = LinearRandom(n_contexts=20, 
                   n_actions=5, 
                   feature_dim=6, 
                   random_state=SEED)
assert is_hls(env.features, env.param)

reps = [env.features]

