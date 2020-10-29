#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:05:20 2020

@author: papini
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator

def moments(dfs):
    cdf = pd.concat(dfs, sort=True).groupby(level=0)
    return cdf.mean(), cdf.std().fillna(0)

def compare(logdir, names, key, seeds):
    os.chdir(logdir)
    
    handles = []
    for name in names:
        dfs = [pd.read_csv(name + '.' + str(seed) + '.csv') for seed in seeds]
        means, stds = moments(dfs)
        mean = means[key]
        std = stds[key]
        xx = np.arange(len(mean))
        line, = plt.plot(xx, mean, label=name)
        plt.fill_between(xx, mean - std, mean + std, alpha=0.3)
        handles.append(line)
        
    plt.xlabel('Iterations')
    plt.ylabel(key)
    plt.legend(handles=handles)
    plt.show()

def plot_ci(logdir, name, key, seeds, rows=None):
    os.chdir(logdir)
    dfs = [pd.read_csv(name + '.' + str(seed) + '.csv') for seed in seeds]
    means, stds = moments(dfs)
    mean = means[key]
    std = stds[key]
    xx = np.arange(len(mean))
    if rows:
        xx = xx[:rows]
        mean = mean[:rows]
        std = std[:rows]
    line, = plt.plot(xx, mean, label=name)
    plt.fill_between(xx, mean - std, mean + std, alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel(key)
    plt.show()
    
def plot_all(logdir, name, key, seeds, rows=None):
    os.chdir(logdir)
    dfs = [pd.read_csv(name + '.' + str(seed) + '.csv') for seed in seeds]
    for df in dfs:
        val = df[key]
        if rows:
            val = val[:rows]
        xx = np.arange(len(val))
        plt.plot(xx, val)

    plt.xlabel('Iterations')
    plt.ylabel(key)
    plt.show()
    
def tournament(logdir, names, key, seeds, rows=None, score='last'):
    os.chdir(logdir)
    
    scores = dict()
    for name in names:
        dfs = [pd.read_csv(name + '.' + str(seed) + '.csv') for seed in seeds]
        means, _ = moments(dfs)
        mean = means[key].to_numpy()
        if rows:
            mean = mean[:rows]
        if score == 'last':
            scores[name] = mean[-1]
        elif score == 'sum':
            scores[name] = sum(mean)
        
    return sorted(scores.items(), key=operator.itemgetter(1))
    
    