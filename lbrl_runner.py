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
import logging

from lbrl.utils import make_synthetic_features, inv_sherman_morrison
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import derank_hls, hls_lambda, is_hls
from lbrl.leader import LEADER
from lbrl.linucb import LinUCB
from lbrl.leaderselect import LEADERSelect
import matplotlib.pyplot as plt


def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="lbrl_leader")
def my_app(cfg: DictConfig) -> None:

    work_dir = Path.cwd()
    original_dir = get_original_cwd()
    log.info(f"Current working directory : {work_dir}")
    log.info(f"Orig working directory    : {original_dir}")

    set_seed_everywhere(cfg.seed)

    ########################################################################
    # Problem creation
    ########################################################################
    ncontexts, narms, dim = cfg.ncontexts, cfg.narms, cfg.dim
    features, theta = make_synthetic_features(
        n_contexts=ncontexts, n_actions=narms, dim=dim,
        context_generation=cfg.contextgeneration, feature_expansion=cfg.feature_expansion,
        seed=cfg.seed_problem
    )

    
    # assert ncontexts == dim
    # features = np.zeros((ncontexts, narms, dim))
    # for i in range(dim):
    #     features[i,0,i] = 1
    #     features[i,1,i+1 if i+1 < dim else 0] = 1 - cfg.mingap
    #     for j in range(2, narms):
    #         features[i,j,:] = (2 * np.random.rand(dim) - 1) / dim
    # theta = np.ones(dim)


    env = LinearEnv(features=features.copy(), param=theta.copy(), rew_noise=cfg.noise_param, random_state=cfg.seed)
    problem_gen = np.random.RandomState(cfg.seed_problem)

    rep_list = []
    param_list = []
    rep_list.append(LinearRepresentation(features))
    param_list.append(theta)
    for i in range(1, dim):
        fi, pi = derank_hls(features=features, param=theta, newrank=i, transform=True, normalize=True, seed=cfg.seed_problem)
        # if np.random.binomial(1, p=0.1):
        #     print(f"adding random noise to rep {i-1}")
        #     fi = fi + np.random.randn(*fi.shape)
        rep_list.append(LinearRepresentation(fi))
        param_list.append(pi)

    true_reward = features @ theta

    # compute gap
    min_gap = np.inf
    for i in range(true_reward.shape[0]):
        rr = true_reward[i]
        sort_rr = sorted(rr)
        gap = sort_rr[-1] - sort_rr[-2]
        min_gap = min(gap, min_gap)
    log.info(f"min gap: {min_gap}")

    # non realizable
    n_nonrealizable = cfg.num_nonrealizable
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


    for i in range(len(rep_list)):
        log.info("\n")
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
