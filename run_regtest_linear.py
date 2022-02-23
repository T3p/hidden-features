import envs as bandits  
import torch
import torch.nn as nn 
from torch.nn import functional as F
from torch.nn.modules import Module
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
import os
from algs.nnmodel import Network


def train_full(test_data, model, learning_rate=1e-2, weight_decay=0, max_epochs=10, batch_size=64, device="cpu", logfolder="",
weight_mse=1,weight_spectral=1, weight_l2features=0):
    writer = SummaryWriter(logfolder)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    batch_counter = 0
    tot_loss = []
    postfix = {
            'loss': 0.0,
            'perc optimal actions': 0.0,
        }
    with tqdm(initial=0, total=max_epochs, postfix=postfix) as pbar:
        batch_counter = 0
        for epoch in range(max_epochs):
            lh = []
            model.train()
            bcnt = 0
            A = 0
            totmse = 0
            totl2feat = 0
            totspectral= 0
            for b_features, b_rewards in test_data:
                # MSE LOSS
                if not np.isclose(weight_mse,0):
                    prediction = model(b_features)
                    mse_loss = F.mse_loss(prediction, b_rewards)
                    totmse = totmse + weight_mse * mse_loss

                #DETERMINANT or LOG_MINEIG LOSS
                if not np.isclose(weight_spectral,0):
                    # I use the fact that the reward is just 0/1
                    idxs_opt = torch.argmax(b_rewards).item()
                    features_optimal_actions = b_features[[idxs_opt]]

                    phi = model.embedding(features_optimal_actions)
                    A = A + torch.sum(phi[...,None]*phi[:,None], axis=0)

                # FEATURES NORM LOSS
                if not np.isclose(weight_l2features,0):
                    phi = model.embedding(b_features)
                    l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
                    # l2 reg on parameters can be done in the optimizer
                    # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
                    totl2feat = totl2feat + weight_l2features * l2feat_loss

                bcnt += 1
                if bcnt % batch_size == 0 or bcnt >= len(test_data):
                    # TOTAL LOSS
                    writer.add_scalar('mse_loss', totmse, batch_counter)
                    if not np.isclose(weight_spectral,0):
                        # det_loss = torch.logdet(A)
                        A = A + 1e-3 * torch.eye(phi.shape[1])
                        spectral_loss = torch.log(torch.linalg.eigvalsh(A).min())
                        totspectral = weight_spectral * spectral_loss
                        writer.add_scalar('spectral_loss',  totspectral, batch_counter)
                    if not np.isclose(weight_l2features,0):
                        writer.add_scalar('l2feat_loss', totl2feat, batch_counter)

                    loss = totmse + totspectral + totl2feat
                    writer.add_scalar('batch_loss', loss, batch_counter)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_counter += 1
                    lh.append(loss.item())
                    A = 0
                    totmse = 0
                    totl2feat = 0
                    totspectral= 0
            writer.add_scalar("epoch_loss", np.mean(lh), epoch)

            model.eval()
            accuracy = 0
            cnt = 0
            for b_features, b_rewards in test_data:
                prediction = model(b_features)
                predicted_optimal_action = torch.argmax(prediction).item()
                optimal_action = torch.argmax(b_rewards).item()
                accuracy += 1 if predicted_optimal_action == optimal_action else 0
                cnt += 1

            writer.add_scalar("epoch percentage optimal actions", accuracy / cnt, epoch)


            if np.mean(lh) < 1e-3:
                break
            tot_loss.append(np.mean(lh))


            postfix['loss'] = tot_loss[-1]
            postfix['perc optimal actions'] = accuracy / cnt
            pbar.set_postfix(postfix)
            pbar.update(1)

    return {
        'loss': tot_loss
    }

### MOVE TO HYDRA FOR MAIN SCRIPT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiClass Bandit Test')
    parser.add_argument('--dim', type=int, default=20, metavar='Context dimension')
    parser.add_argument('--narms', type=int, default=5, metavar='Number of actions')
    parser.add_argument('--horizon', type=int, default=10000, metavar='Horizon of the bandit problem, corresponding to horizon x nactions samples ')
    parser.add_argument('--seed', type=int, default=0, metavar='Seed used for the generation of the bandit problem')
    parser.add_argument('--bandittype', default="onehot", help="None, expanded, onehot")
    parser.add_argument('--contextgeneration', default="uniform", help="uniform, gaussian, bernoulli")
    parser.add_argument('--layers', nargs='+', type=int, default=100, help="dimension of each layer (example --layers 100 200)")
    parser.add_argument('--logfolder', type=str, default="tblogs")
    parser.add_argument('--max_epochs', type=int, default=250, help="maximum number of epochs")
    parser.add_argument('--batch_size', type=int, default=200, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay")
    parser.add_argument('--config_name', type=str, default="", help='configuration name used to create the log')
    parser.add_argument('--weight_mse', type=float, default=1, help="weight_mse")
    parser.add_argument('--weight_spectral', type=float, default=0, help="weight_spectral")
    parser.add_argument('--weight_l2features', type=float, default=0, help="weight_l2features")
    parser.add_argument('--device', type=str, default="cpu", help="PyTorch device")

    args = parser.parse_args()
    env = bandits.LinearContinuous(
        context_dim=args.dim, num_actions=args.narms, context_generation=args.contextgeneration,
        feature_expansion=args.bandittype, seed=args.seed
    )
    hid_dim = args.layers
    if not isinstance(args.layers, list):
        hid_dim = [args.layers]
    layers = [(el, nn.ReLU()) for el in hid_dim]
    net = Network(env.feature_dim, layers)
    print(net)

    print(f'Input features dim: {env.feature_dim}')

    # train dataset
    X, Y = None, None
    test_data = []
    for i in range(args.horizon):
        # with this we get a matrix (na x ndim) for x and (na) for y
        env.sample_context()
        x = env.features()
        y = x @ env.theta
        test_data.append((torch.FloatTensor(x,device=args.device),torch.FloatTensor(y.reshape(-1,1),device=args.device)))
        if X is None:
            X = x 
            Y = y
        X = np.concatenate((X,x), axis=0)
        Y = np.concatenate((Y,y), axis=0)
    print(f"Features (expanded): {X.shape}")
    print(X.shape)
    print(Y.shape)

    train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float, device=args.device),
                torch.tensor(Y.reshape(-1,1), dtype=torch.float, device=args.device)
                )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    log_path = f"tblogs/linear_{args.bandittype}{args.config_name}"
    isExist = os.path.exists(log_path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(log_path)


    config = vars(args)
    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(config,f, indent=4, sort_keys=True)

    net.to(args.device)
    results = train_full(
        # train_loader=train_loader, 
        test_data=test_data, model=net, 
        learning_rate=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        device=args.device,
        logfolder=log_path,
        weight_spectral=args.weight_spectral,
        weight_mse=args.weight_mse, weight_l2features=args.weight_l2features
    )
    plt.plot(results['loss'])
    plt.savefig(os.path.join(log_path, "loss.png"))


