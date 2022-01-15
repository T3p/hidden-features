from __future__ import print_function
import argparse
from scipy.sparse.linalg import interface
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import dump


class Net(nn.Module):
    def __init__(self, n_input, fit_bias=True):
        super(Net, self).__init__()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(n_input, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1, bias=fit_bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc3(x)
        return output
    
    def features(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--file', type=str, default="jester_svd36.npz",
                        help='dataset (default: jester_svd36.npz)')
    parser.add_argument('--model', type=str, default="mlp",
                        help='model (default: mlp)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--fit_intercept', action='store_true', default=False,
                        help='Fit also intercept in the linear model')
    parser.add_argument('--sample_nfeatures', type=int, default=-1, metavar='N',
                        help='sample randomly n features (default: -1 (no sampling)')
    args = parser.parse_args()

    data = np.load(args.file)
    X = data['X']
    y = data['y']
    user_features = data['user_features']
    arm_features = data['arm_features']

    sub_sampling = ""
    if args.sample_nfeatures > 0:
        # np.random.seed(args.seed)
        dim = X.shape[1]
        sub_dim = min(args.sample_nfeatures, dim)
        print(f"Sampling {sub_dim} features out of {dim}: ")
        idxs = np.random.choice(dim, size=sub_dim, replace=False)
        print(idxs)
        X = X[:, idxs]
        sub_sampling = f"-s{sub_dim}"

    print(X.shape)
        
    name = f"jester_{args.model}{sub_sampling}.p"
    name_predict = f"jester_{args.model}{sub_sampling}_fitted.npz"
    if args.model == "mlp":
        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {'batch_size': args.batch_size}
        # test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {
                # 'num_workers': 1,
                        # 'pin_memory': True,
                        'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            # test_kwargs.update(cuda_kwargs)

        torch_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float, device=device), torch.tensor(y.reshape(-1,1), dtype=torch.float, device=device)
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=torch_dataset,
            **train_kwargs
        )

        model = Net(n_input=X.shape[1], fit_bias=args.fit_intercept).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        # optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            # test(model, device, test_loader)
            scheduler.step()

        with torch.no_grad():
            y_pred = model(torch.tensor(X, dtype=torch.float, device=device)).cpu().detach().numpy()
            score = r2_score(y, y_pred)
            print(f"score= {score}")
            features = model.features(torch.tensor(X, dtype=torch.float, device=device)).cpu().detach().numpy()

        if args.save_model:
            torch.save(model.state_dict(), f"jester_{args.model}{sub_sampling}.pt")
    elif args.model == "linear":
        model = LinearRegression(fit_intercept=args.fit_intercept)
        print("started training Linear Regressor")
        start_time = time.time()
        model.fit(X, y)
        end_time = time.time()
        print(f"--- {end_time - start_time} seconds ---")
        print(f"score= {model.score(X, y)}")
        print(model.intercept_)
        if args.save_model:
            dump(model, name)
        y_pred = model.predict(X)

        features = X
        if args.fit_intercept:
            features = np.concatenate([X, np.ones((X.shape[0],1))], axis=1)

    model = LinearRegression(fit_intercept=args.fit_intercept)
    model.fit(features, y_pred)
    print(f"linear score= {model.score(X, y)}")


    y_pred = y_pred.reshape(user_features.shape[0], arm_features.shape[0])
    features = features.reshape(user_features.shape[0], arm_features.shape[0], X.shape[1])
    np.savez_compressed(name_predict, features=features, predictions=y_pred)

    # w = np.concatenate( [model.coef_.ravel(), np.array([model.intercept_])] )
    # A = features @ w
    # assert np.allclose(A, y_pred)


if __name__ == '__main__':
    main()
