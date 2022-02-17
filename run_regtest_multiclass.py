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

class Network(nn.Module):

    def __init__(self, input_size:int, layers_data:list):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.embedding_dim = layers_data[-1][0]
        self.fc2 = nn.Linear(self.embedding_dim, 1, bias=False)
    
    def embedding(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        return self.fc2(x)


def train_full(train_loader, test_data, model, learning_rate=1e-2, weight_decay=0, max_epochs=10, batch_size=64, device="cpu", logfolder="",
weight_mse=1, weight_spectral=1, weight_l2features=0):
    writer = SummaryWriter(logfolder)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
            for b_features, b_rewards in train_loader:
                loss = 0
                N = b_features.shape[0]
                # MSE LOSS
                if not np.isclose(weight_mse,0):
                    prediction = model(b_features)
                    mse_loss = F.mse_loss(prediction, b_rewards)
                    writer.add_scalar('mse_loss', weight_mse * mse_loss, batch_counter)
                    loss = loss + weight_mse * mse_loss

                #DETERMINANT or LOG_MINEIG LOSS
                if not np.isclose(weight_spectral,0):
                    phi = model.embedding(b_features)
                    A = torch.sum(phi[...,None]*phi[:,None], axis=0) + 1e-3 * torch.eye(phi.shape[1])
                    # det_loss = torch.logdet(A)
                    spectral_loss = torch.log(torch.linalg.eigvalsh(A).min()/N)
                    writer.add_scalar('spectral_loss',  weight_spectral * spectral_loss, batch_counter)
                    loss = loss + weight_spectral * spectral_loss

                # FEATURES NORM LOSS
                if not np.isclose(weight_l2features,0):
                    l2feat_loss = torch.sum(torch.norm(phi, p=2, dim=1))
                    # l2 reg on parameters can be done in the optimizer
                    # though weight_decay (https://discuss.pytorch.org/t/simple-l2-regularization/139)
                    writer.add_scalar('l2feat_loss', weight_l2features * l2feat_loss, batch_counter)
                    loss = loss + weight_l2features * l2feat_loss

                # TOTAL LOSS
                writer.add_scalar('batch_loss', loss, batch_counter)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_counter += 1
                lh.append(loss.item())
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
    parser.add_argument('--dataset', default='magic', metavar='DATASET')
    parser.add_argument('--bandittype', default='expanded', metavar='DATASET', help="expanded or num")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--layers', nargs='+', type=int, default=100, help="dimension of each layer (example --layers 100 200)")
    parser.add_argument('--logfolder', type=str, default="tblogs")
    parser.add_argument('--max_epochs', type=int, default=250, help="maximum number of epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay")
    parser.add_argument('--config_name', type=str, default="", help='configuration name used to create the log')
    parser.add_argument('--weight_mse', type=float, default=1, help="weight_mse")
    parser.add_argument('--weight_spectral', type=float, default=0, help="weight_spectral")
    parser.add_argument('--weight_l2features', type=float, default=0, help="weight_l2features")
    parser.add_argument('--device', type=str, default="cpu", help="PyTorch device")

    args = parser.parse_args()
    env = bandits.make_from_dataset(
        args.dataset, bandit_model=args.bandittype)
    print(f"Samples: {env.X.shape}")
    print(f'Labels: {np.unique(env.y)}')
    print('layers: ', args.layers)
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
    for i in range(len(env)):
        # with this we get a matrix (na x ndim) for x and (na) for y 
        x, y = env.__getitem__(i)
        test_data.append((torch.FloatTensor(x,device=args.device),torch.FloatTensor(y,device=args.device)))
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
    
    log_path = f"tblogs/{args.dataset}_{args.bandittype}{args.config_name}"
    isExist = os.path.exists(log_path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(log_path)


    config = vars(args)
    with open(os.path.join(log_path, "config.json"), "w") as f:
        json.dump(config,f, indent=4, sort_keys=True)

    net.to(args.device)
    results = train_full(
        train_loader=train_loader, test_data=test_data, model=net, 
        learning_rate=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        device=args.device,
        logfolder=log_path,
        weight_mse=args.weight_mse, weight_spectral=args.weight_spectral, weight_l2features=args.weight_l2features
    )
    plt.plot(results['loss'])
    plt.savefig(os.path.join(log_path, "loss.png"))


