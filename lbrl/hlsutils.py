import pdb
import numpy as np


"""Computes minimum eigenvalue of AA^T"""
def min_eig_outer(A, weak=False):
    EPS = 1e-8
    _, sv, _ = np.linalg.svd(A)
    i = 0
    if weak:
        while abs(sv[len(sv)-1-i])<EPS and i<len(sv):
            i += 1
            
    return sv[len(sv)-1-i]**2

def optimal_arms(rewards):
    return np.argmax(rewards, axis=1)

def optimal_rewards(features, rewards):
    n_contexts = features.shape[0]
    ii = np.arange(n_contexts)
    return rewards[ii, optimal_arms(rewards)]

def optimal_features(features, rewards):
    n_contexts = features.shape[0]
    ii = np.arange(n_contexts)
    return features[ii, optimal_arms(rewards), :]


#Diversity properties    
def rank(features, rewards, tol=None):
    n_contexts, n_arms, dim = features.shape
    all_feats = np.reshape(features, 
                           (n_contexts * n_arms, dim))
    D = np.matmul(all_feats.transpose(1,0), all_feats)
    return np.linalg.matrix_rank(D, tol)

def spans(features, rewards, tol=None):
    n_contexts, n_arms, dim = features.shape
    return rank(features, rewards, tol) == dim

def is_cmb(features, rewards, tol=None):
    n_contexts, n_arms, dim = features.shape
    for a in range(n_arms):
        feats = features[:, a, :]
        if np.linalg.matrix_rank(feats, tol) < dim:
            return False
    return True

def cmb_rank(features, rewards, tol=None):
    n_contexts, n_arms, dim = features.shape
    min_rnk = dim
    for a in range(n_arms):
        feats = features[:, a, :]
        rnk = np.linalg.matrix_rank(feats, tol)
        if rnk == 0:
            return 0
        if rnk < min_rnk:
            min_rnk = rnk
    return min_rnk

def hls_rank(features, rewards, tol=None):
    phi = optimal_features(features, rewards)
    D  = np.matmul(phi.transpose(1,0), phi)
    return np.linalg.matrix_rank(D, tol)

def is_hls(features, rewards, tol=None):
    dim = features.shape[2]
    return hls_rank(features, rewards, tol) == dim

def hls_lambda(features, rewards, cprobs=None, weak=False):

    phi = optimal_features(features, rewards)
    D  = np.matmul(phi.transpose(1,0), phi)
    n2 = np.linalg.eigvalsh(D).min() / phi.shape[0]
    return n2

def normalize_linrep(features, param, scale=1.):
    param_norm = np.linalg.norm(param)
    new_param = param / param_norm * scale
    new_features = features * param_norm / scale
    return new_features, new_param

def random_transform(features, param, normalize=True, seed=0):
    rng = np.random.RandomState(seed)
    dim = len(param)

    A = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(A)
        
    new_features = features @ q
    new_param = q.T @ param
            
    if normalize:
        new_features, new_param = normalize_linrep(new_features, new_param)
    
    assert np.allclose(features @ param, new_features @ new_param)
    return new_features, new_param

def derank_hls(features, param, newrank=1, transform=True, normalize=True, seed=0):
    nc = features.shape[0]

    rewards = features @ param #SxA
    # compute optimal arms
    opt_arms = np.argmax(rewards, axis=1) #S
    
    # compute features of optimal arms
    opt_feats = features[np.arange(nc), opt_arms, :] #SxD
    opt_rews = rewards[np.arange(nc), opt_arms].reshape((nc, 1)) #Sx1
    remove = min(max(nc - newrank + 1, 0), nc)
    
    new_features = np.array(features, dtype=np.float32)
    outer = np.outer(opt_rews[:remove], opt_rews[:remove])
    xx = outer @ opt_feats[:remove, :] \
        / np.linalg.norm(opt_rews[:remove], 2)**2
    new_features[np.arange(remove), opt_arms[:remove], :] = xx
    new_param = param.copy()
    
    if transform:
        new_features, new_param = random_transform(new_features, new_param, normalize=normalize, seed=seed)
    elif normalize:
        new_features, new_param = normalize_linrep(new_features, new_param)
        
    # assert np.allclose(features @ param, new_features @ new_param)

    return new_features, new_param


def reduce_dim(features, param, newdim, transform=True, normalize=True, seed=0):
    assert newdim <= param.shape[0] and newdim > 0
    f1 = features.copy().astype(np.float64)
    p1 = param.copy().astype(np.float64)
    dim = param.shape[0]
    
    for _ in range(dim - newdim):
        f1[:, :, 1] = f1[:, :, 0] * p1[0] + f1[:, :, 1] * p1[1]
        p1[1] = 1.
        f1 = f1[:, :, 1:]
        p1 = p1[1:]    

    if transform:
        f1, p1 = random_transform(f1, p1, normalize=normalize, seed=seed)
    elif normalize:
        f1, p1 = normalize_linrep(f1, p1)

    rewards = features @ param
    new_rewards = f1 @ p1
    assert np.allclose(rewards, new_rewards)
    assert p1.shape[0] == newdim
    return f1, p1
