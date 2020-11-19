#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:01:06 2020

@author: papini
"""
from lrcb.bandits.finite_linear_bandits import FiniteLinearBandit

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
        self._current_id = default
    
    def select_rep(self, i):
        self._features = self.reps[i].features
        self._param = self.reps[i]._param
        self.dim = self.reps[i].dim
        self._current_id = i
    
    def rep(self):
        return self.reps[self._current_id]
    
    def rep_id(self):
        return self._current_id
        


    
    
    
    
    