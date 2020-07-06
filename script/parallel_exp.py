#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:18:56 2019

@author: matteo
"""

import argparse
import screenutils as su

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--command', type=str, default='echo hello')
parser.add_argument('--name', type=str, default='hello')
args = parser.parse_args()

seeds = [8448, 170, 2007, 716, 953, 3282, 8437, 9993, 8615, 3306]

for seed in seeds:    
    screen = su.Screen(args.name + '_' + str(seed), initialize=True)
    commands = args.command + ' --seed %d' % seed
    screen.send_commands(commands)
