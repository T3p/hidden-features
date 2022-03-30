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
    return np.linalg.matrix_rank(all_feats, tol)

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
    return np.linalg.matrix_rank(optimal_features(features, rewards), tol)

def is_hls(features, rewards, tol=None):
    dim = features.shape[2]
    return hls_rank(features, rewards, tol) == dim

def hls_lambda(features, rewards, cprobs=None, weak=False):
    n_contexts = features.shape[0]
    if cprobs is None:
        mineig = min_eig_outer(optimal_features(features, rewards), weak) / n_contexts
    else:
        assert np.allclose(np.sum(cprobs), 1.)
        mineig = min_eig_outer(np.sqrt(np.array(cprobs)[:, None]) * 
                               optimal_features(features, rewards), weak)
    if np.allclose(mineig, 0.):
        return 0.
    return mineig


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
        
    assert np.allclose(features @ param, new_features @ new_param)

    return new_features, new_param


if __name__=="__main__":
    seed = 1234
    rng = np.random.RandomState(seed=seed)
    nc = 2
    na = 2
    d = 2
    #features = rng.binomial(1, 0.5, size=(nc, na, d))
    features = rng.normal(0, 1, size=(nc, na, d))
    param = rng.uniform(-1., 1., size=d)
    rewards = features @ param

    assert is_hls(features, rewards)
    
    
    f1, p1 = derank_hls(features, param, newrank=1, transform=True, normalize=False, seed=seed)
    print(hls_rank(f1, f1@p1, tol=1e-6))
