import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as normalize_matrix

def features_rank(features, tol=None):
    all_feats = np.reshape(features, 
                           (features.shape[0] * features.shape[1], features.shape[2]))
    return np.linalg.matrix_rank(all_feats, tol)

def hls_rank(features, param, tol=None):
    nc = features.shape[0]
    rewards = features @ param
    # compute optimal arms
    opt_arms = np.argmax(rewards, axis=1)
    # compute features of optimal arms
    opt_feats = features[np.arange(nc), opt_arms, :]
    return np.linalg.matrix_rank(opt_feats, tol)

def is_hls(features, param, tol=None):
    return hls_rank(features, param, tol) == len(param)

def pred_error(features, param, rewards):
    pred_rew = features @ param
    err = np.abs(pred_rew-rewards)
    return np.min(err), np.max(err)

#Transforming representations
def normalize_linrep(features, param, scale=1.):
    param_norm = np.linalg.norm(param)
    new_param = param / param_norm * scale
    new_features = features * param_norm / scale
    return new_features, new_param

def random_transform(features, param, normalize=True, seed=0):
    rng = np.random.RandomState(seed)
    dim = len(param)
    A = rng.normal(size=(dim, dim))

    q, r = np.linalg.qr(A) #q is an orthogonal matrix
    
    new_features = features @ q
    new_param = q.T @ param
        
    if normalize:
        new_features, new_param = normalize_linrep(new_features, new_param)
        
    assert np.allclose(features @ param, new_features @ new_param)
    return new_features, new_param

def make_random_linrep(
    n_contexts, n_actions, feature_dim, 
    ortho=True, normalize=True, seed=0,
    method="gaussian"):

    rng = np.random.RandomState(seed)
    if method == "gaussian":
        features = rng.normal(size=(n_contexts, n_actions, feature_dim))
    elif method == "bernoulli":
        features = rng.binomial(n=1, p=rng.rand(), size=(n_contexts, n_actions, feature_dim))

    param = 2 * rng.uniform(size=feature_dim) - 1
    
    #Orthogonalize features
    if ortho:
        features = np.reshape(features, (n_contexts * n_actions, feature_dim))
        orthogonalizer = PCA(n_components=feature_dim, random_state=seed) #no dimensionality reduction
        features = orthogonalizer.fit_transform(features)
        features = np.reshape(features, (n_contexts, n_actions, feature_dim))
        features = np.take(features, rng.permutation(feature_dim), axis=2)
    
    if normalize:
        features, param = normalize_linrep(features, param)
        
    return features, param


def derank_hls(features, param, newrank=1, transform=True, normalize=True, seed=0):
    nc = features.shape[0]

    rewards = features @ param
    # compute optimal arms
    opt_arms = np.argmax(rewards, axis=1)
    # compute features of optimal arms
    opt_feats = features[np.arange(nc), opt_arms, :]
    opt_rews = rewards[np.arange(nc), opt_arms].reshape((nc, 1)) 
    remove = min(max(nc - newrank + 1, 0), nc)
    
    new_features = np.array(features)
    outer = np.matmul(opt_rews[:remove], opt_rews[:remove].T)
    xx = np.matmul(outer, opt_feats[:remove, :]) \
        / np.linalg.norm(opt_rews[:remove])**2
    new_features[np.arange(remove), opt_arms[:remove], :] = xx
    
    new_param = param.copy()
    
    if transform:
        new_features, new_param = random_transform(new_features, new_param, normalize=normalize, seed=seed)
    elif normalize:
        new_features, new_param = normalize_linrep(new_features, new_param)
        
    assert np.allclose(features @ param, new_features @ new_param)
    return new_features, new_param

def reduce_dim(features, param, newdim=2, transform=True, normalize=True, seed=0):
    dim = features.shape[-1]
    assert newdim <= dim and newdim > 0
    f1 = np.array(features)
    p1 = np.array(param)
    
    for _ in range(dim - newdim):
        f1[:, :, 1] = f1[:, :, 0] * p1[0] + f1[:, :, 1] * p1[1]
        p1[1] = 1.
        f1 = f1[:, :, 1:]
        p1 = p1[1:]
        
    if transform:
        f2, p2 = random_transform(f1, p1, normalize=normalize, seed=seed)
    elif normalize:
        f2, p2 = normalize_linrep(f1, p1)
    else: 
        f2, p2 = f1, p1
    
    assert f2.shape[-1] == newdim
    return f2, p2

def fuse_columns(features, param, cols, transform=True, normalize=True, seed=0):
    dim = features.shape[-1]
    n_contexts = features.shape[0]
    rewards = features @ param
    opt_arms = np.argmax(rewards, axis=1)
    f1 = np.array(features)
    for i in range(dim):
        f1[:, :, i] *= param[i]
    p1 = 1.0 + 0.0 * np.array(param)
    ss = np.arange(n_contexts)

    v = np.zeros(n_contexts)
    for i in cols:
        assert i >= 0 and i < dim
        v += f1[ss, opt_arms, i] / len(cols)

    for i in cols:
        f1[ss, opt_arms, i] = v
        
    if transform:
        f2, p2 = random_transform(f1, p1, normalize=normalize, seed=seed)
    elif normalize:
        f2, p2 = normalize_linrep(f1, p1)
    else: 
        f2, p2 = f1, p1

    return f2, p2

def make_hls_rank(rewards, dim, rank, transform=True, normalize=True, eps=0.1, seed=0):
    opt_rewards = np.max(rewards, axis=1)
    if np.allclose(np.linalg.norm(opt_rewards), 0.):
        eps = eps / 2.
        rewards = rewards + eps
        opt_rewards = np.max(rewards, axis=1)
    opt_arms = np.argmax(rewards, axis=1)
    nc, na = rewards.shape
    
    param = np.zeros(dim)
    param[0] = 1
    sup = np.max(np.abs(opt_rewards)) + eps
    features, _ = make_random_linrep(nc, na, dim)
    opt_features = np.zeros((nc, dim))
    for i in range(1, rank):
        opt_features[i, i] = sup
    features[np.arange(nc), opt_arms, :] = opt_features
    features[:, :, 0] = rewards
    
    if transform:
        f1, p1 = random_transform(features, param, normalize=normalize, seed=seed)
    elif normalize:
        f1, p1 = normalize_linrep(features, param)
    else:
        f1, p1 = features, param
    
    assert np.allclose(f1 @ p1, rewards)
    return f1, p1