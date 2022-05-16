#!/bin/bash

HORIZON=200000
SEEDS=123,694,386,238,944,874,245,2,56
ALGO=nnlinucb
# nnlinucb no reg, glrt yes,false
python run_linearfinite.py -m horizon=${HORIZON} algo=${ALGO} check_glrt=True,False layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} epsilon_decay=none device="cuda" use_tb=false use_wandb=true bonus_scale=1 domain=wheel.yaml

# BanditSRL Linucb
python run_linearfinite.py -m horizon=${HORIZON} algo=${ALGO} check_glrt=True layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=1 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} epsilon_decay=none device="cuda" use_tb=false use_wandb=true bonus_scale=1 use_maxnorm=True,False domain=wheel.yaml
python run_linearfinite.py -m horizon=${HORIZON} algo=${ALGO} check_glrt=True layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=1 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=0 weight_spectral=0 seed=${SEEDS} epsilon_decay=none device="cuda" use_tb=false use_wandb=true bonus_scale=1 use_maxnorm=True,False domain=wheel.yaml
python run_linearfinite.py -m horizon=${HORIZON} algo=${ALGO} check_glrt=True layers=\"100,100,50,20,10\" weight_mse=1 weight_rayleigh=0 weight_min_features=0 weight_min_random=0 weight_l2features=0 weight_trace=1 weight_spectral=0 seed=${SEEDS} epsilon_decay=none device="cuda" use_tb=false use_wandb=true bonus_scale=1 use_maxnorm=False domain=wheel.yaml
