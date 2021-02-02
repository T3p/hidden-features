import numpy as np
import os
from scipy.sparse.linalg import svds
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from lrcb.utils import check_spanrd

"""
Preprocess jester dataset to create a linear contextual bandit problem
The dataset can be found in

Carlos Riquelme, George Tucker, Jasper Snoek:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling. 
ICLR (Poster) 2018

Link: https://storage.googleapis.com/bandits_datasets/jester_data_40jokes_19181users.npy
"""

output_folder = "./"
input_file = os.path.join(output_folder, 'jester_data_40jokes_19181users.npy')


def preprocess(K=36, hidden=32, test_size=0.25, normalize=False):

    ratings = np.load(input_file)
    n_users, n_items = ratings.shape
    print("Loaded dataset: {}".format(ratings.shape))
    ratings = ratings / 10

    U, s, Vt = svds(ratings, k=K)
    s = np.diag(s)
    U = np.dot(U, s)

    # MSE
    Yhat = U.dot(Vt)
    rmse = np.sqrt(np.mean(np.abs(Yhat - ratings) ** 2))
    print("K[SVDS]: ", K)
    print("RMSE[SVDS]:", rmse)
    print("MAX_ERR[SVDS]:", np.abs(Yhat - ratings).max())

    X, y = [], []
    for t in range(n_users):
        for z in range(n_items):
            feat = np.concatenate([U[t], Vt[:, z]]).ravel()
            X.append(feat)
            y.append(ratings[t, z])
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print("Training NN -- Size {0}".format((hidden, hidden)))
    regr = MLPRegressor(hidden_layer_sizes=(hidden, hidden), max_iter=500, verbose=True).fit(X_train, y_train)

    print("R^2:", regr.score(X_test, y_test))

    # Build features
    X_pred = X

    hidden_layer_sizes = list(regr.hidden_layer_sizes)

    layer_units = [X_pred.shape[1]] + hidden_layer_sizes + [1]
    activations = [X_pred]
    for i in range(regr.n_layers_ - 1):
        activations.append(np.empty((X_pred.shape[0], layer_units[i + 1])))

    regr._forward_pass(activations)
    y_pred = activations[-1]
    print("MSE (original):", np.mean((y_pred.flatten() - y) ** 2))

    # get weights
    last_w = regr.coefs_[-1]
    bias = np.array(regr.intercepts_[-1]).reshape((1, 1))
    last_w = np.concatenate([last_w, bias])

    # get last-layer features
    last_feat = np.array(activations[-2], dtype=np.float32)
    last_feat = np.concatenate([last_feat, np.ones((X_pred.shape[0], 1))], axis=1)

    # get prediction
    pred = last_feat.dot(last_w)
    print("MSE (recomputed with last layer only):", np.mean((pred.flatten() - y) ** 2))

    # get feature matrix
    d = hidden_layer_sizes[-1] + 1
    print("d={0}".format(d))
    phi = np.empty((n_users, n_items, d), dtype=np.float32)
    idx = 0
    for t in range(n_users):
        for z in range(n_items):
            phi[t, z, :] = last_feat[idx, :] / (np.linalg.norm(last_feat[idx, :]) if normalize else 1)
            idx += 1
    assert idx == last_feat.shape[0]

    # get param
    theta = np.array(last_w, dtype=np.float32).squeeze()
    if normalize:
        theta = theta / np.linalg.norm(theta)

    # check span
    mu = phi.dot(theta)
    astar = np.argmax(mu, axis=1)
    fstar = np.array([phi[x, astar[x]] for x in range(n_users)])

    span = d
    for i in range(d):
        if check_spanrd(fstar, d - i):
            span = d - i
            break

    print("{0}Spanning R^{1}".format("WARNING: " if span == d else "", span))

    if not os.path.exists(output_folder):
        try:
            os.mkdir(output_folder)
        except OSError:
            print("Creation of the directory {} failed".format(output_folder))
        else:
            print("Successfully created the directory {}".format(output_folder))
    np.savez_compressed(os.path.join(output_folder, 'jester_nn_d{0}_span{1}.npz'.format(d,span)),
                        ratings=y, features=phi, theta=theta)

    return idx, X, y, theta


if __name__ == '__main__':
    for hidden in [16, 32, 64, 128]:
        preprocess(hidden=hidden, K=36, normalize=True, test_size=0.25)
