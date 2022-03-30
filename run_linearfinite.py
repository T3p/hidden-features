from turtle import pd
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import ConvertMode


import os
from pathlib import Path

import numpy as np
import torch
import random

import xbrl.envs as bandits
import xbrl.envs.hlsutils as hlsutils
from xbrl.algs.linear import LinUCB
from xbrl.algs.batched.nnlinucb import NNLinUCB
from xbrl.algs.batched.nnleader import NNLeader
import pickle
import json
from xbrl.algs.nnmodel import MLLinearNetwork, MLLogisticNetwork
import torch.nn as nn

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')

    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    features, theta = bandits.make_synthetic_features(
        n_contexts=cfg.domain.ncontexts, n_actions=cfg.domain.narms, dim=cfg.domain.dim,
        context_generation=cfg.domain.contextgeneration, feature_expansion=cfg.domain.feature_expansion,
        seed=cfg.domain.seed_problem
    )
    rewards = features @ theta
    print(f"Original rep -> HLS rank: {hlsutils.hls_rank(features, rewards)} / {features.shape[2]}")
    print(f"Original rep -> is HLS: {hlsutils.is_hls(features, rewards)}")
    print(f"Original rep -> HLS min eig: {hlsutils.hls_lambda(features, rewards)}")
    print(f"Original rep -> is CMB: {hlsutils.is_cmb(features, rewards)}")
    if cfg.domain.newrank not in [None, "none", "None"]:
        features, theta = hlsutils.derank_hls(features=features, param=theta, newrank=cfg.domain.newrank)
        rewards = features @ theta
        print(f"New rep -> HLS rank: {hlsutils.hls_rank(features, rewards)} / {features.shape[2]}")
        print(f"New rep -> is HLS: {hlsutils.is_hls(features, rewards)}")
        print(f"New rep -> HLS min eig: {hlsutils.hls_lambda(features, rewards)}")
        print(f"New rep -> is CMB: {hlsutils.is_cmb(features, rewards)}")

    env = bandits.CBFinite(
        feature_matrix=features, 
        rewards=rewards, seed=cfg.seed, 
        noise=cfg.domain.noise_type, noise_param=cfg.domain.noise_param
    )
    print(f"min gap: {env.min_suboptimality_gap()}")

    if not cfg.algo == "linucb":
        print('layers: ', cfg.layers)
        if cfg.layers not in [None, "none", "None"]:
            hid_dim = cfg.layers
            if not isinstance(cfg.layers, list):
                hid_dim = [cfg.layers]
            layers = [(el, nn.Tanh()) for el in hid_dim]
        else:
            layers = None # linear in the features
        net = MLLinearNetwork(env.feature_dim, layers).to(cfg.device)
        print(net)

    if cfg.algo == "nnlinucb":
        algo = NNLinUCB(
            env=env,
            model=net,
            device=cfg.device,
            batch_size=cfg.batch_size,
            max_updates=cfg.max_updates,
            update_every_n_steps=cfg.update_every_n_steps,
            learning_rate=cfg.lr,
            buffer_capacity=cfg.buffer_capacity,
            noise_std=cfg.noise_std,
            delta=cfg.delta,
            weight_decay=cfg.weight_decay,
            ucb_regularizer=cfg.ucb_regularizer,
            bonus_scale=cfg.bonus_scale,
            reset_model_at_train=cfg.reset_model_at_train
        )
    elif cfg.algo == "linucb":
        algo = LinUCB(
            env=env,
            seed=cfg.seed,
            update_every_n_steps=cfg.update_every_n_steps,
            noise_std=cfg.noise_std,
            delta=cfg.delta,
            ucb_regularizer=cfg.ucb_regularizer,
            bonus_scale=cfg.bonus_scale
        )
    elif cfg.algo == "nnleader":
        algo = NNLeader(
            env=env,
            model=net,
            device=cfg.device,
            batch_size=cfg.batch_size,
            max_updates=cfg.max_updates,
            update_every_n_steps=cfg.update_every_n_steps,
            learning_rate=cfg.lr,
            buffer_capacity=cfg.buffer_capacity,
            noise_std=cfg.noise_std,
            delta=cfg.delta,
            weight_decay=cfg.weight_decay,
            ucb_regularizer=cfg.ucb_regularizer,
            bonus_scale=cfg.bonus_scale,
            reset_model_at_train=cfg.reset_model_at_train,
            weight_mse=cfg.weight_mse,
            weight_spectral=cfg.weight_spectral,
            weight_l2features=cfg.weight_l2features,
            weight_orth=cfg.weight_orth
        )
    else:
        raise ValueError("Unknown algorithm {cfg.algo}")
    print(type(algo).__name__)
    algo.reset()
    result = algo.run(horizon=cfg.horizon, log_path=work_dir)#cfg.log_dir)

    # if cfg.save_dir is not None:
    #     with open(os.path.join(cfg.save_dir, "config.json"), 'w') as f:
    #         json.dump(OmegaConf.to_yaml(cfg), f)
    #     with open(os.path.join(cfg.save_dir, "result.pkl"), 'wb') as f:
    #         pickle.dump(result, f)
    

    with open(os.path.join(work_dir, "config.json"), 'w') as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=4, sort_keys=True)
    with open(os.path.join(work_dir, "result.pkl"), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    my_app()