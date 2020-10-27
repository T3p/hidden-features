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