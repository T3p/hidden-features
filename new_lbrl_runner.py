from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path, get_original_cwd

import os
from pathlib import Path
import numpy as np
import random
import pickle
import json
import logging
import wandb

from lbrl.utils import make_synthetic_features, inv_sherman_morrison
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import derank_hls, hls_lambda, is_hls, hls_rank
from lbrl.leader import LEADER
import lbrl.superreplearner as SRL
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

    if cfg.use_wandb:
        # wandb.tensorboard.patch(root_logdir=str(work_dir))
        wandb.init(
            # Set the project where this run will be logged
            project="lbrl", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{cfg.exp_name}", 
            # Track hyperparameters and run metadata
            config=OmegaConf.to_container(cfg),
            # sync_tensorboard=True
        )

    ########################################################################
    # Problem creation
    ########################################################################
    if cfg.domain.type == "finite_single":
        ncontexts, narms, dim = cfg.domain.ncontexts, cfg.domain.narms, cfg.domain.dim
        features, theta = make_synthetic_features(
            n_contexts=ncontexts, n_actions=narms, dim=dim,
            context_generation=cfg.domain.contextgeneration, feature_expansion=cfg.domain.feature_expansion,
            seed=cfg.domain.seed_problem
        )

        env = LinearEnv(features=features.copy(), param=theta.copy(), rew_noise=cfg.domain.noise_param, random_state=cfg.seed)
        true_reward = features @ theta
        problem_gen = np.random.RandomState(cfg.domain.seed_problem)
        rep_list = [LinearRepresentation(features)]
        param_list = [theta]
        cfg.rep_idx = 0
        position_reference_rep = 0
    elif cfg.domain.type == "finite_multi":
        ncontexts, narms, dim = cfg.domain.ncontexts, cfg.domain.narms, cfg.domain.dim
        features, theta = make_synthetic_features(
            n_contexts=ncontexts, n_actions=narms, dim=dim,
            context_generation=cfg.domain.contextgeneration, feature_expansion=cfg.domain.feature_expansion,
            seed=cfg.domain.seed_problem
        )

        env = LinearEnv(features=features.copy(), param=theta.copy(), rew_noise=cfg.domain.noise_param, random_state=cfg.seed)
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
        env = LinearEnv(features=features_list[position_reference_rep].copy(), param=param_list[position_reference_rep].copy(), rew_noise=cfg.domain.noise_param, random_state=cfg.seed)
        true_reward = features_list[position_reference_rep] @ param_list[position_reference_rep]
        problem_gen = np.random.RandomState(cfg.domain.seed_problem)
        rep_list = []
        assert len(features_list) == len(param_list)
        for i in range(len(features_list)):
            rep_list.append(LinearRepresentation(features_list[i]))
        del features_list

    # compute gap
    min_gap = np.inf
    min_gap_ctx = []
    na = true_reward.shape[1]
    for ctx in range(true_reward.shape[0]):
        rr = true_reward[ctx]
        arr = sorted(rr)
        for i in range(na-1):
            diff = arr[i+1] - arr[i]
            if diff <= min_gap and diff > 0:
                min_gap = diff
                min_gap_ctx.append(ctx)

    log.info(f"min gap: {min_gap}")
    log.info(f"min gap [rewards]: {true_reward[min_gap_ctx]}")

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
        log.info(f"is HLS({i}): {is_hls(rep_list[i].features, true_reward, tol=1e-6)}")
        log.info(f"HSL rank({i}): {hls_rank(rep_list[i].features, true_reward, tol=1e-6)}")
        log.info(f"lambda HLS({i}): {hls_lambda(rep_list[i].features, true_reward)}")
    log.info("\n")

    del true_reward

    M = len(rep_list)
    if cfg.algo == "linucb":
        cfg.check_glrt = False
        algo = SRL.SRLLinUCB(env=env, representations=[rep_list[cfg.rep_idx]], 
            features_bounds = [np.linalg.norm(rep_list[cfg.rep_idx].features, 2, axis=-1).max()],
            param_bounds=[np.linalg.norm(param_list[cfg.rep_idx], 2)],
            cfg=cfg
        )
    elif cfg.algo == "egreedyglrt":
        algo = SRL.SRLEGreedy(env=env, representations=[rep_list[cfg.rep_idx]],
            features_bounds = [np.linalg.norm(rep_list[cfg.rep_idx].features, 2, axis=-1).max()],
            param_bounds=[np.linalg.norm(param_list[cfg.rep_idx], 2)],
            cfg=cfg
        )
    elif cfg.algo == "leader_old":
        algo = LEADER(env, representations=rep_list, reg_val=cfg.reg_val, noise_std=cfg.noise_std, 
                features_bounds=[np.linalg.norm(rep_list[j].features, 2, axis=-1).max() for j in range(M)], 
                param_bounds=[np.linalg.norm(param_list[j], 2) for j in range(M)],
                random_state=cfg.seed, delta=cfg.delta
            )
    elif cfg.algo == "leader":
        cfg.check_glrt = False
        algo = SRL.Leader(env, representations=rep_list, 
                features_bounds=[np.linalg.norm(rep_list[j].features,2, axis=-1).max() for j in range(M)], 
                param_bounds=[np.linalg.norm(param_list[j],2) for j in range(M)],
                cfg=cfg
            )
    elif cfg.algo.startswith("srl"):
        if cfg.algo.endswith("mineig"):
            select_method = SRL.SuperRepLearner.MINEIG
        elif cfg.algo.endswith("mineig_norm"):
            select_method = SRL.SuperRepLearner.MINEIG_NORM
        elif cfg.algo.endswith("avg_quad"):
            select_method = SRL.SuperRepLearner.AVG_QUAD
        elif cfg.algo.endswith("avg_quad_norm"):
            select_method = SRL.SuperRepLearner.AVG_QUAD_NORM
        else:
            raise ValueError(f"unknown algo {cfg.algo}")
        cfg.select_method = select_method
        if cfg.algo.startswith("srllinucb"):
            algo = SRL.SRLLinUCB(env, representations=rep_list, 
                features_bounds=[np.linalg.norm(rep_list[j].features,2, axis=-1).max() for j in range(M)], 
                param_bounds=[np.linalg.norm(param_list[j],2) for j in range(M)],
                cfg=cfg
            )
        if cfg.algo.startswith("srlegreedy"):
            algo = SRL.SRLEGreedy(env, representations=rep_list, 
                features_bounds=[np.linalg.norm(rep_list[j].features,2, axis=-1).max() for j in range(M)], 
                param_bounds=[np.linalg.norm(param_list[j],2) for j in range(M)],
                cfg=cfg
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

    if cfg.use_wandb:
        wandb.finish(quiet=True)


if __name__ == "__main__":
    my_app()
