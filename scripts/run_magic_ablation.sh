#!/bin/bash

HORIZON=200000
# SEEDS=713,464,777,879,660,608,773,919,591,229,727,184,307,662,600,770,442,150,336,52
SEEDS=713,464,777,879
ALGO=nnegreedy

# MSE
python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=True hydra.sweep.dir=magic_0517/\${algo}_mse
# Rayleigh (maxnorm)
python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=1 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=True hydra.sweep.dir=magic_0517/\${algo}_rayleigh_maxnorm
# Rayleigh
python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=1 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=False hydra.sweep.dir=magic_0517/\${algo}_rayleigh
# Min-features (maxnorm)
python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=True hydra.sweep.dir=magic_0517/\${algo}_minfeat_maxnorm
# Min-features
python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=False hydra.sweep.dir=magic_0517/\${algo}_minfeat
# Min-random (maxnorm)
# python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=1 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=True hydra.sweep.dir=magic_0517/\${algo}_minrand_maxnorm
# Min-random
# python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=1 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=False hydra.sweep.dir=magic_0517/\${algo}_minrand
# Trace (maxnorm)
# python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=1 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=True hydra.sweep.dir=magic_0517/\${algo}_trace_maxnorm
# Trace
# python run_linearfinite.py -m domain=magic.yaml horizon=${HORIZON} algo=${ALGO} check_glrt=True glrt_scale=5 bonus_scale=3 layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=1 weight_spectral=0 seed=${SEEDS} use_tb=true use_maxnorm=False hydra.sweep.dir=magic_0517/\${algo}_trace