#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:01:06 2020

@author: papini
"""
from lrcb.bandits.finite_linear_bandits import FiniteLinearBandit
import numpy as np

class FiniteMultiBandit(FiniteLinearBandit):
    def __init__(self, n_contexts, n_arms, reps, noise=0.1,
                 context_probs=None, default=0):
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.reps = reps
        self.dim = self.reps[default].dim
        self._features = self.reps[default].features
        self._param = self.reps[default]._param
        self._noise = noise
        if context_probs is not None:
            assert len(context_probs) == self.n_contexts
            assert np.allclose(sum(context_probs), 1.)
        self._cprobs = context_probs
        
        self._context = None
        self._current_id = default
    
    def select_rep(self, i):
        self._features = self.reps[i].features
        self._param = self.reps[i]._param
        self.dim = self.reps[i].dim
        self._current_id = i
        
    def reset(self):
        i = np.random.choice(len(self.reps))
        self.select_rep(i)
    
    def rep(self):
        return self.reps[self._current_id]
    
    def rep_id(self):
        return self._current_id


def hls_rank_combined(multibandit, tol=None):
    feats = [r._optimal_features() for r in multibandit.reps]
    combined = np.concatenate(feats, axis=1)
    return np.linalg.matrix_rank(combined, tol)
    
    
    
    
    