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

def optimal_arms(rewards, tol=1e-5):
    rows, cols = np.where( (rewards.max(axis=1).reshape(-1,1) - rewards) < tol )
    # return np.argmax(rewards, axis=1)
    return rows, cols

def optimal_rewards(features, rewards):
    n_contexts = features.shape[0]
    ii = np.arange(n_contexts)
    opt_arms = np.argmax(rewards, axis=1)
    return rewards[ii, opt_arms]

def optimal_features(features, rewards):
    # n_contexts = features.shape[0]
    # ii = np.arange(n_contexts)
    # return features[ii, optimal_arms(rewards), :]
    rows, cols = optimal_arms(rewards)
    return features[rows, cols]

#Diversity properties    
def rank(features, tol=None):
    n_contexts, n_arms, dim = features.shape
    all_feats = np.reshape(features, 
                           (n_contexts * n_arms, dim))
    A = np.matmul(all_feats.T, all_feats)
    return np.linalg.matrix_rank(A, tol, hermitian=True)

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
    A  = np.matmul(phi.T, phi) / phi.shape[0]
    return np.linalg.matrix_rank(A, tol, hermitian=True)

def is_hls(features, rewards, tol=None):
    dim = features.shape[2]
    return hls_rank(features, rewards, tol) == dim

def hls_lambda(features, rewards, cprobs=None, weak=False):

    phi = optimal_features(features, rewards)
    A  = np.matmul(phi.T, phi) / phi.shape[0]
    eig = np.linalg.eigvalsh(A).min()
    return eig

    # n_contexts = features.shape[0]
    # if cprobs is None:
    #     mineig = min_eig_outer(optimal_features(features, rewards), weak) / n_contexts
    # else:
    #     assert np.allclose(np.sum(cprobs), 1.)
    #     mineig = min_eig_outer(np.sqrt(np.array(cprobs)[:, None]) * 
    #                            optimal_features(features, rewards), weak)
    # if np.allclose(mineig, 0.):
    #     return 0.
    # assert np.isclose(mineig,n2,atol=1e-3), (mineig,n2)



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

    # compute optimal arms
    rows, cols = optimal_arms(rewards=rewards)
    opt_feats = features[rows, cols]
    nel = len(rows)

    # compute features of optimal arms
    opt_rews = rewards[rows, cols]
    remove = min(max(nel - newrank + 1, 0), nel)
    
    new_features = np.array(features, dtype=np.float32)
    outer = np.outer(opt_rews[:remove], opt_rews[:remove])
    xx = outer @ opt_feats[:remove, :] \
        / np.linalg.norm(opt_rews[:remove], 2)**2
    new_features[rows[:remove], cols[:remove], :] = xx
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
