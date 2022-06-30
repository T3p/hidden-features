import pdb
from omegaconf import DictConfig, OmegaConf
import omegaconf
import hydra
from hydra.utils import to_absolute_path, get_original_cwd
import os
from pathlib import Path
import numpy as np
import torch
import random
from xbrl import TORCH_FLOAT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import xbrl.envs as bandits
from xbrl.algs.nnmodel import MLLinearNetwork, initialize_weights
import torch.nn as nn
from torch.nn import functional as F
import logging
import matplotlib.pyplot as plt
import json

# A logger for this file
log = logging.getLogger(__name__)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="conf/xbrl", config_name="regconfig.yaml")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    work_dir = Path.cwd()
    original_dir = get_original_cwd()
    print(f"Current working directory : {work_dir}")
    print(f"Orig working directory    : {original_dir}")
    with open(os.path.join(work_dir, "config.json"), 'w') as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=4, sort_keys=True)

    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    if cfg.domain.type == "dataset":
        env = bandits.make_from_dataset(
            cfg.domain.dataset, bandit_model=cfg.domain.bandittype, 
            rew_optimal=cfg.domain.rew_optimal, rew_suboptimal=cfg.domain.rew_suboptimal,
            seed=cfg.seed, noise=cfg.domain.noise_type, noise_param=cfg.domain.noise_param)

        print()
        print("="*20)
        print(env.description())
        print("="*20,"\n")
        log.info("="*20)
        log.info(env.description())
        log.info("="*20)

    d = env.feature_dim
    nc = env.X.shape[0]
    na = env.action_space.n
    X = np.zeros((nc*na, d))
    Y = np.zeros(nc*na)
    test_data = []
    for i in range(nc):
        # get features and rewards in the current context
        fc, rc = env.__getitem__(i)
        X[i*na:(i+1)*na, :] = fc
        Y[i*na:(i+1)*na] = rc
        test_data.append((torch.FloatTensor(fc,device=device),torch.FloatTensor(rc,device=device)))
    
    train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float, device=device),
                torch.tensor(Y.reshape(-1,1), dtype=torch.float, device=device)
                )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)

    print('layers: ', cfg.layers)
    print(type(cfg.layers))
    if cfg.layers not in [None, "none", "None"]:
        hid_dim = cfg.layers
        if isinstance(cfg.layers, str):
            print(cfg.layers.split(","))
            hid_dim = cfg.layers.split(",")
            hid_dim = [int(el) for el in hid_dim]
        if not (isinstance(hid_dim, list) or isinstance(hid_dim, omegaconf.listconfig.ListConfig)):
            hid_dim = [hid_dim]
        print(hid_dim)
        layers = [(el, nn.ReLU() if cfg.use_relu else nn.LeakyReLU()) for el in hid_dim]
    else:
        layers = None # linear in the features
    net = MLLinearNetwork(env.feature_dim, layers).to(device)
    initialize_weights(net)

    log_path = work_dir
    writer = SummaryWriter(log_path)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    net.train()

    batch_counter = 0
    tot_loss = []
    tot_accuracy = []
    postfix = {
            'loss': 0.0,
            'perc optimal actions': 0.0,
        }
    with tqdm(initial=0, total=cfg.max_epochs, postfix=postfix) as pbar:
        batch_counter = 0
        for epoch in range(cfg.max_epochs):
            lh = []
            net.train()
            for b_features, b_rewards in train_loader:
                loss = 0
                N = b_features.shape[0]
                prediction = net(b_features)
                mse_loss = F.mse_loss(prediction, b_rewards)
                writer.add_scalar('mse_loss', mse_loss, batch_counter)
                loss = loss + mse_loss

                # TOTAL LOSS
                writer.add_scalar('batch_loss', loss, batch_counter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_counter += 1
                lh.append(loss.item())
            writer.add_scalar("epoch_loss", np.mean(lh), epoch)

            net.eval()
            accuracy = 0
            cnt = 0
            for b_features, b_rewards in test_data:
                prediction = net(b_features)
                predicted_optimal_action = torch.argmax(prediction).item()
                optimal_action = torch.argmax(b_rewards).item()
                accuracy += 1 if predicted_optimal_action == optimal_action else 0
                cnt += 1

            writer.add_scalar("epoch percentage optimal actions", accuracy / cnt, epoch)
            tot_accuracy.append(accuracy / cnt)


            if np.mean(lh) < 1e-3:
                break
            tot_loss.append(np.mean(lh))


            postfix['loss'] = tot_loss[-1]
            postfix['perc optimal actions'] = accuracy / cnt
            pbar.set_postfix(postfix)
            pbar.update(1)

    plt.figure()
    plt.plot(tot_loss)
    plt.title("train mse")
    plt.savefig(os.path.join(log_path, "loss.png"))
    plt.figure()
    plt.plot(tot_accuracy)
    plt.title("perc of optimal actions (greedy)")
    plt.savefig(os.path.join(log_path, "accuracy.png"))

if __name__ == "__main__":
    my_app()