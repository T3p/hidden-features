import openml
import numpy as np
import random
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
import argparse

def fetch_dataset(id):
    ds = openml.datasets.get_dataset(id)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    
    return X, y

def standardize(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)

def representation_dataset(X,y):
    X = standardize(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    assert len(y) == n_samples
    classes = y.values.unique()
    n_classes = len(classes)
    
    new_X = np.zeros((n_samples*n_classes, n_features+n_classes))
    new_y = np.zeros(n_samples*n_classes)
    
    for i in range(n_samples):
        for j in range(n_classes):
            one_hot = np.zeros(n_classes)
            one_hot[j] = 1.
            new_X[i*n_classes + j] = np.concatenate((X[i], one_hot))
            new_y[i*n_classes + j] = 1. if y[i] == classes[j] else 0.
    return new_X, new_y

def build_model(net, X, n_contexts, n_actions):
    
    # Build features
    hidden_layer_sizes = list(net.hidden_layer_sizes)

    layer_units = [X.shape[1]] + hidden_layer_sizes + [1]
    activations = [X]
    for i in range(net.n_layers_ - 1):
        activations.append(np.empty((X.shape[0], layer_units[i + 1])))

    net._forward_pass(activations)

    # get weights
    last_w = net.coefs_[-1]
    bias = np.array(net.intercepts_[-1]).reshape((1, 1))
    last_w = np.concatenate([last_w, bias])

    # get last-layer features
    last_feat = np.array(activations[-2], dtype=np.float32)
    last_feat = np.concatenate([last_feat, np.ones((X.shape[0], 1))], axis=1)

    # get feature matrix
    d = hidden_layer_sizes[-1] + 1
    phi = np.empty((n_contexts, n_actions, d), dtype=np.float32)
    idx = 0
    for t in range(n_contexts):
        for z in range(n_actions):
            phi[t, z, :] = last_feat[idx, :] 
            idx += 1
    assert idx == last_feat.shape[0]

    # get param
    theta = np.array(last_w, dtype=np.float32).squeeze()

    return phi, theta

def learn_representation(X, y, test_size=0.25, hidden=(32,32), max_iter=500, 
                         mode='regression'):
    n_contexts = X.shape[0]
    n_actions = len(y.values.unique())
    
    #Preprocess dataset 
    X_rep, y_rep = representation_dataset(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_rep, y_rep, test_size=test_size)
    
    #Fit NN
    if mode=='regression':
        net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                           verbose=False)
    elif mode=='classification':
        net = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, 
                           verbose=False)
    else: raise ValueError
    
    net.fit(X_train, y_train)
    score = net.score(X_test, y_test)
    
    #Build representation
    phi, theta = build_model(net, X_rep, n_contexts, n_actions)
    return phi, theta, score

if __name__ == '__main__':    
    ids = list(np.load('ids.npy'))
    
    parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='regression')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--n_datasets', type=int, default=len(ids))
    parser.add_argument('--path', type=str, default='../problem_data/openml/')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    hidden=(args.hidden, args.hidden)

    count = 0
    for id in ids[:args.n_datasets]:
        i = int(id)
        count += 1
        X, y = fetch_dataset(i)
        phi, theta, score = learn_representation(X, y, args.test_size, hidden, args.max_iter, mode=args.mode)
        print('%d/%d (ID=%d): score=%f' % (count, args.n_datasets, i , score))
        np.savez_compressed(args.path+'openml_{0}_id{1}_dim{2}_seed{3}.npz'.format(args.mode,
                                i, args.hidden+1, args.seed), 
                        features=phi, theta=theta, score=score)