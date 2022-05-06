import pdb
from turtle import pd
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path, get_original_cwd


import os
from pathlib import Path

import numpy as np
import torch
import random
from xbrl import TORCH_FLOAT

import xbrl.envs as bandits
import xbrl.envs.hlsutils as hlsutils
# from xbrl.algs.linear import LinUCB
from xbrl.algs.batched.nnlinucb import NNLinUCB
from xbrl.algs.batched.linucb import LinUCB
from xbrl.algs.batched.nnepsilongreedy import NNEpsGreedy
import xbrl.algs.incremental as incalg
# import xbrl.algs.nnleaderinc as incalg
import pickle
import json
from xbrl.algs.nnmodel import MLLinearNetwork, MLLogisticNetwork, initialize_weights
import torch.nn as nn
import copy
import wandb
import logging

# A logger for this file
log = logging.getLogger(__name__)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="conf/xbrl", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    work_dir = Path.cwd()
    original_dir = get_original_cwd()
    print(f"Current working directory : {work_dir}")
    print(f"Orig working directory    : {original_dir}")

    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)


    if cfg.use_wandb:
        exp_name = '_'.join([
            cfg.algo, cfg.domain.type
        ])
        wandb.init(
            # Set the project where this run will be logged
            project="run_linearfinite",
            group=cfg.algo,   # mode="disabled",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=exp_name,
            # Track hyperparameters and run metadata
            config=OmegaConf.to_container(cfg)
        )

    ########################################################################
    # Problem creation
    ########################################################################
    
    if cfg.domain.type == "finite" or cfg.domain.type == "toy":
        if cfg.domain.type == "finite":
            features, theta = bandits.make_synthetic_features(
                n_contexts=cfg.domain.ncontexts, n_actions=cfg.domain.narms, dim=cfg.domain.dim,
                context_generation=cfg.domain.contextgeneration, feature_expansion=cfg.domain.feature_expansion,
                seed=cfg.domain.seed_problem
            )
        else:
            ncontexts, narms, dim = cfg.domain.ncontexts, cfg.domain.narms, cfg.domain.dim
            assert ncontexts == dim
            features = np.zeros((ncontexts, narms, dim))
            for i in range(dim):
                features[i,0,i] = 1
                features[i,1,i+1 if i+1 < dim else 0] = 1 - cfg.domain.mingap
                for j in range(2, narms):
                    features[i,j,:] = (2 * np.random.rand(dim) - 1) / dim
            theta = np.ones(dim)
        
        rewards = features @ theta
        print(f"Original rep -> HLS rank: {hlsutils.hls_rank(features, rewards)} / {features.shape[2]}")
        print(f"Original rep -> is HLS: {hlsutils.is_hls(features, rewards)}")
        print(f"Original rep -> HLS min eig: {hlsutils.hls_lambda(features, rewards)}")
        print(f"Original rep -> HLS rank: {hlsutils.hls_rank(features, rewards)}")
        print(f"Original rep -> is CMB: {hlsutils.is_cmb(features, rewards)}")
        if cfg.domain.type == "finite" and cfg.domain.newrank not in [None, "none", "None"]:
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
    elif cfg.domain.type == "fromfile":
        print(os.path.exists(os.path.join(original_dir, cfg.domain.datafile)))
        if os.path.exists(os.path.join(original_dir, cfg.domain.datafile)):
            features_list, param_list, position_reference_rep = np.load(os.path.join(original_dir, cfg.domain.datafile), allow_pickle=True)
        else:
            print(cfg.domain.url)
            if cfg.domain.url is not None:
                print('-'*20)
                print(f"please download the file using the following link: {cfg.domain.url}")
                print(f"and save it into : {os.path.join(original_dir, cfg.domain.datafile)}")
                print('-'*20)
                exit(9)
            else:
                raise ValueError(f"Unable to open the file {cfg.domain.datafile}")
        features = features_list[position_reference_rep]
        theta = param_list[position_reference_rep]
        rewards = features @ theta
        print(f"Original rep -> HLS rank: {hlsutils.hls_rank(features, rewards)} / {features.shape[2]}")
        print(f"Original rep -> is HLS: {hlsutils.is_hls(features, rewards)}")
        print(f"Original rep -> HLS min eig: {hlsutils.hls_lambda(features, rewards)}")
        print(f"Original rep -> HLS rank: {hlsutils.hls_rank(features, rewards)}")
        print(f"Original rep -> is CMB: {hlsutils.is_cmb(features, rewards)}")

        env = bandits.CBFinite(
            feature_matrix=features, 
            rewards=rewards, seed=cfg.seed, shuffle=False,
            noise=cfg.domain.noise_type, noise_param=cfg.domain.noise_param
        )
        print(f"min gap: {env.min_suboptimality_gap()}")

    elif cfg.domain.type == "nn":
        net_file = os.path.join(original_dir, cfg.domain.net)
        features_file = os.path.join(original_dir, cfg.domain.features)
        print(net_file, features_file)
        model = torch.load(net_file)
        model.eval()
        with open(features_file, 'rb') as f:
            features = np.load(f)
        ncontexts, narms, dim = features.shape
        xt = torch.tensor(features.reshape(-1, dim), dtype=torch.float)
        rewards = model(xt).detach().numpy().ravel()
        rewards = rewards.reshape(ncontexts, narms)
        del xt
        
        env = bandits.CBFinite(
            feature_matrix=features, 
            rewards=rewards, seed=cfg.seed, 
            noise=cfg.domain.noise_type, noise_param=cfg.domain.noise_param
        )
        print(f"min gap: {env.min_suboptimality_gap()}")
    else:
        raise ValueError(f"Unknown domain type {cfg.domain.type}")


    if not cfg.algo == "linucb":
        if cfg.domain.type == "nn":
            net = copy.deepcopy(model).to(device)
        else:
            print('layers: ', cfg.layers)
            print(type(cfg.layers))
            if cfg.layers not in [None, "none", "None"]:
                hid_dim = cfg.layers
                if isinstance(cfg.layers, str):
                    print(cfg.layers.split(","))
                    hid_dim = cfg.layers.split(",")
                    hid_dim = [int(el) for el in hid_dim]
                if not isinstance(hid_dim, list):
                    hid_dim = [hid_dim]
                print(hid_dim)
                layers = [(el, nn.Tanh()) for el in hid_dim]
            else:
                layers = None # linear in the features
            net = MLLinearNetwork(env.feature_dim, layers).to(device)
        
        if cfg.random_init_weights:
            print("randomly initializing weights of algorithm network")
            initialize_weights(net)
        print(net)

    if cfg.algo == "nnlinucb":
        assert cfg.weight_spectral == 0 or cfg.weight_rayleigh == 0 or cfg.weight_orth == 0
        algo = NNLinUCB(env, cfg, net)
    elif cfg.algo == "nnleader":
        assert cfg.weight_spectral > 0 or cfg.weight_rayleigh > 0 or cfg.weight_orth > 0
        algo = NNLinUCB(env, cfg, net)
    elif cfg.algo == "nnegreedy":
        algo = NNEpsGreedy(env, cfg, net)
    elif cfg.algo == "egreedy":
        algo = NNEpsGreedy(env, cfg)
    elif cfg.algo == "linucb":
        algo = NNLinUCB(env, cfg)
    elif cfg.algo == "nnlinucbinc":
        algo = incalg.NNLinUCBInc(env, cfg, net)
    elif cfg.algo == "nnleaderinc":
        algo = incalg.NNLeaderInc(env, cfg, net)
    elif cfg.algo == "nneginc":
        algo = incalg.NNEGInc(env, cfg, net)
    else:
        raise ValueError("Unknown algorithm {cfg.algo}")
    print(type(algo).__name__)
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
    payload = {'model': algo.model, 'features': algo.env.feature_matrix, 'rewards': algo.env.rewards}
    with open(os.path.join(work_dir, "algo.pt"), 'wb') as f:
        torch.save(payload, f)
    
    if cfg.use_wandb:
        wandb.finish(quiet=True)


if __name__ == "__main__":
    # torch.set_default_dtype(TORCH_FLOAT)
    my_app()