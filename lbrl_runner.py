from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path, get_original_cwd

import os
from pathlib import Path
import numpy as np
import random
import pickle
import json
import copy
import requests
import logging

from lbrl.utils import make_synthetic_features, inv_sherman_morrison
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import derank_hls, hls_lambda, is_hls
from lbrl.leader import LEADER
from lbrl.linucb import LinUCB
from lbrl.leaderselect import LEADERSelect
from lbrl.leaderselectlb import LEADERSelectLB
import matplotlib.pyplot as plt


def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path="conf/lbrl", config_name="conf")
def my_app(cfg: DictConfig) -> None:

    work_dir = Path.cwd()
    original_dir = get_original_cwd()
    log.info(f"Current working directory : {work_dir}")
    log.info(f"Orig working directory    : {original_dir}")

    set_seed_everywhere(cfg.seed)

    ########################################################################
    # Problem creation
    ########################################################################
    if cfg.domain.type == "finite":
        ncontexts, narms, dim = cfg.ncontexts, cfg.narms, cfg.dim
        features, theta = make_synthetic_features(
            n_contexts=ncontexts, n_actions=narms, dim=dim,
            context_generation=cfg.contextgeneration, feature_expansion=cfg.feature_expansion,
            seed=cfg.domain.seed_problem
        )

        env = LinearEnv(features=features.copy(), param=theta.copy(), rew_noise=cfg.noise_param, random_state=cfg.seed)
        true_reward = features @ theta
        problem_gen = np.random.RandomState(cfg.domain.seed_problem)


        rep_list = []
        param_list = []
        rep_list.append(LinearRepresentation(features))
        param_list.append(theta)
        position_reference_rep = 0
        for i in range(1, dim+1):
            fi, pi = derank_hls(features=features, param=theta, newrank=i, transform=True, normalize=True, seed=cfg.domain.seed_problem)
            # if np.random.binomial(1, p=0.1):
            #     print(f"adding random noise to rep {i-1}")
            #     fi = fi + np.random.randn(*fi.shape)
            rep_list.append(LinearRepresentation(fi))
            param_list.append(pi)

        # non realizable
        n_nonrealizable = cfg.domain.num_nonrealizable
        for i in range(n_nonrealizable):
            idx = problem_gen.choice(len(rep_list), 1).item()
            fi = rep_list[idx].features
            # mask = np.random.binomial(1, p=0.5, size=fi.shape)
            fi = fi + problem_gen.randn(*fi.shape) * 0.6
            rep_list.append(LinearRepresentation(fi))
            mtx, bv = np.eye(fi.shape[2])/0.0001, 0
            for kk in range(fi.shape[0]):
                for aa in range(fi.shape[1]):
                    el = fi[kk,aa]
                    mtx, _ = inv_sherman_morrison(el, mtx)
                    bv = bv + true_reward[kk,aa] * el
            pi = mtx @ bv
            param_list.append(pi) # best fit to the true reward

    
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
        env = LinearEnv(features=features_list[position_reference_rep].copy(), param=param_list[position_reference_rep].copy(), rew_noise=cfg.noise_param, random_state=cfg.seed)
        true_reward = features_list[position_reference_rep] @ param_list[position_reference_rep]
        problem_gen = np.random.RandomState(cfg.domain.seed_problem)
        rep_list = []
        assert len(features_list) == len(param_list)
        for i in range(len(features_list)):
            rep_list.append(LinearRepresentation(features_list[i]))
        del features_list

    # compute gap
    min_gap = np.inf
    for i in range(true_reward.shape[0]):
        rr = true_reward[i]
        sort_rr = sorted(rr)
        gap = sort_rr[-1] - sort_rr[-2]
        min_gap = min(gap, min_gap)
    log.info(f"min gap: {min_gap}")

    for i in range(len(rep_list)):
        log.info("\n")
        if i == position_reference_rep:
            log.info(f"Info representation({i}) [REFERENCE REP]")
        else:
            log.info(f"Info representation({i})")
        log.info(f"dim({i}): {rep_list[i].features_dim()}")
        log.info(f"feature norm({i}): {np.linalg.norm(rep_list[i].features,2,axis=-1).max()}")
        log.info(f"param norm({i}): {np.linalg.norm(param_list[i],2)}")
        current_reward = rep_list[i].features @ param_list[i]
        error = np.abs(current_reward - true_reward).max()
        log.info(f"min gap: {min_gap}")
        log.info(f"realizable({i}): {error < min_gap}")
        log.info(f"error({i}): {error}")
        log.info(f"is HLS({i}): {is_hls(rep_list[i].features, true_reward)}")
        log.info(f"lambda HLS({i}): {hls_lambda(rep_list[i].features, true_reward)}")
    log.info("\n")

    del true_reward

    M = len(rep_list)
    if cfg.algo == "linucb":
        algo = LinUCB(env, representation=rep_list[cfg.linucb_rep], reg_val=cfg.ucb_regularizer, noise_std=cfg.noise_param, 
                features_bound=np.linalg.norm(env.features, 2, axis=-1).max(),
                param_bound=np.linalg.norm(env.param, 2),
                random_state=cfg.seed, delta=cfg.delta
            )
    elif cfg.algo == "leader":
        algo = LEADER(env, representations=rep_list, reg_val=cfg.ucb_regularizer, noise_std=cfg.noise_param, 
                features_bounds=[np.linalg.norm(rep_list[j].features, 2, axis=-1).max() for j in range(M)], 
                param_bounds=[np.linalg.norm(param_list[j], 2) for j in range(M)],
                check_elim_condition_every=cfg.check_every,
                random_state=cfg.seed, delta=cfg.delta
            )
    elif cfg.algo == "leaderselect":
        algo = LEADERSelect(env, representations=rep_list, reg_val=cfg.ucb_regularizer, noise_std=cfg.noise_param, 
                features_bounds=[np.linalg.norm(rep_list[j].features,2, axis=-1).max() for j in range(M)], 
                param_bounds=[np.linalg.norm(param_list[j],2) for j in range(M)],
                recompute_every=cfg.check_every, normalize=cfg.normalize_mineig,
                random_state=cfg.seed, delta=cfg.delta
        )
    elif cfg.algo == "leaderselectlb":
        algo = LEADERSelectLB(env, representations=rep_list, reg_val=cfg.ucb_regularizer, noise_std=cfg.noise_param, 
                features_bounds=[np.linalg.norm(rep_list[j].features,2, axis=-1).max() for j in range(M)], 
                param_bounds=[np.linalg.norm(param_list[j],2) for j in range(M)],
                recompute_every=cfg.check_every, normalize=cfg.normalize_mineig,
                random_state=cfg.seed, delta=cfg.delta
        )
    else:
        raise ValueError("Unknown algorithm {cfg.algo}")
    
    with open(os.path.join(work_dir, "config.json"), 'w') as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=4, sort_keys=True)

    result = algo.run(cfg.horizon, log_path=work_dir)
    plt.figure(figsize=(10,6))
    plt.plot(result['regret'],label=cfg.algo)
    plt.legend()
    plt.savefig(os.path.join(work_dir, "regret.png"))

    with open(os.path.join(work_dir, "result.pkl"), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    my_app()
