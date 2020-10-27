#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:01:06 2020

@author: papini
"""
import numpy as np
from lrcb.bandits.finite_linear_bandits import FiniteLinearBandit
from sklearn.decomposition import PCA

class LinearRepresentation:
    def __init__(self, features, param):
        self.features = features
        self.n_contexts, self.n_arms, self.dim = features.shape
        assert self.dim == len(param)
        self._param = param
        
        self._rewards = np.matmul(self.features, self._param)
        self._optimal_arms = np.argmax(self._rewards, axis=1)
        ii = np.arange(self.n_contexts)
        self._optimal_rewards = self._rewards[ii, self._optimal_arms]
        self._optimal_features = self.features[ii, self._optimal_arms, :]
        
    def __eq__(self, other):
        return np.allclose(self._rewards, other._rewards)
    
    
def hls_rank(rep, tol=None):
    return np.linalg.matrix_rank(rep._optimal_features, tol)
    
def rank(rep, tol=None):
    all_feats = np.reshape(rep.features, 
                           (rep.n_contexts * rep.n_arms, rep.dim))
    return np.linalg.matrix_rank(all_feats, tol)

class FiniteMultiBandit(FiniteLinearBandit):
    def __init__(self, n_contexts, n_arms, reps, noise=0.1, default=0):
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.reps = reps
        self.dim = self.reps[default].dim
        self._features = self.reps[default].features
        self._param = self.reps[default]._param
        self._noise = noise
        
        self._context = None
    
    def select_rep(self, i):
        self._features = self.reps[i].features
        self._param = self.reps[i]._param
        self.dim = self.reps[i].dim
        

def make_canon_rep(n_contexts, n_arms):
    dim = n_contexts * n_arms
    features = np.eye(dim)
    features = np.reshape(features, (n_contexts, n_arms, dim))
    param = 2 * np.random.uniform(size=dim) - 1
    return LinearRepresentation(features, param)

def make_random_rep(n_contexts, n_arms, dim, ortho=True):
    features = np.random.normal(size=(n_contexts, n_arms, dim))
    param = 2 * np.random.uniform(size=dim) - 1
    
    
    #Orthogonalize features
    if ortho:
        features = np.reshape(features, (n_contexts * n_arms, dim))
        orthogonalizer = PCA(n_components=dim) #no dimensionality reduction
        features = orthogonalizer.fit_transform(features)
        features = np.reshape(features, (n_contexts, n_arms, dim))
        features = np.take(features, np.random.permutation(dim), axis=2)
    
    return LinearRepresentation(features, param)

def derank(rep, newrank=1):
    f0 = rep.features
    opt_feats = rep._optimal_features
    opt_arms = rep._optimal_arms
    nc = rep.n_contexts
    opt_rews = rep._optimal_rewards.reshape((nc, 1)) 
    remove = min(max(nc - newrank + 1, 0), nc)
    
    f1 = np.array(f0)
    outer = np.matmul(opt_rews[:remove], opt_rews[:remove].T)
    xx = np.matmul(outer, opt_feats[:remove, :]) \
        / np.linalg.norm(opt_rews[:remove])**2
    f1[np.arange(remove), opt_arms[:remove], :] = xx
    lr = LinearRepresentation(f1, np.array(rep._param))
    return lr

def fix_rank(rep, newrank, transform=True, eps=0.1):
    nc = rep.n_contexts
    na = rep.n_arms
    dim = max(newrank, rep.dim)
    rewards = np.array(rep._rewards)
    opt_rewards = rep._optimal_rewards
    param = np.zeros(dim)
    param[0] = 1
    sup = np.max(opt_rewards) + eps
    features = np.zeros((nc, na, dim))
    features[:, :, 0] = rewards
    for i in range(1, newrank):
        features[i, :, i] = sup
    
    if transform:
        A = np.random.normal(size=(dim, dim))
        features = np.matmul(features, A)
        param = np.linalg.solve(A, param)
        maxp = np.max(np.abs(param))
        param = param / maxp
        features = features * maxp
    
    return LinearRepresentation(features, param)

if __name__ == '__main__':
    #Destructive approach
    r = make_canon_rep(3, 1)
    r1 = derank(r)
    r2 = derank(r, 2)
    r3 = derank(r, 3)
    
    assert r == r1
    assert r == r2
    assert r == r3
    
    assert(hls_rank(r)) == 3
    assert(hls_rank(r1)) == 1
    assert(hls_rank(r2)) == 2
    assert(hls_rank(r3)) == 3
    
    #Constructive approach
    r = make_random_rep(10, 4, 3)
    assert hls_rank(r) == 3
    r1 = fix_rank(r, 1)
    assert r1 == r
    assert hls_rank(r1) == 1
    r2 = fix_rank(r, 2)
    assert r2 == r
    assert hls_rank(r2) == 2
    r3 = fix_rank(r, 3)
    assert r3 == r
    assert hls_rank(r3) == 3
    r4 = fix_rank(r, 4)
    assert r4 == r
    assert hls_rank(r4) == 4
    
    
    
    
    