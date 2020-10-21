import numpy as np
from lrcb.utils import inverse_norm
from lrcb.logger import Logger
import scipy.sparse.linalg as sparse
from lrcb.utils import feature_basis, basis_completion

"""
    Agnostic approach: just use apparent features (misspecified model)
"""
def oful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, seed=0):    
    logname = 'oracle' if bandit.hidden == 0 else 'oful'
    logname = logname + '.%d.%d.%d.%d' % (bandit.n_contexts, bandit.n_arms, bandit.dim, bandit.hidden)
    logger = Logger(directory='../logs', name = logname + '.' + str (seed), modes=['human', 'csv'])
    log_keys = ['pseudoregret',
                 'cumpseudoregret',
                 'regret', 
                 'cumregret', 
                 'paramerror', 
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_pseudoregret = 0
    cum_regret = 0
    
    dim = bandit.dim - bandit.hidden
    A = reg * np.eye(dim)
    b = np.zeros(dim)
    
    for t in range(horizon):
        #Latest parameter estimate
        param = np.linalg.solve(A, b)
        
        #Observe context
        s = bandit.observe()
        
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            feat = bandit.feat(s, i)
            beta = oful_coeff(A, reg, noise, delta, param_bound)
            bonus = beta * inverse_norm(feat, A)
            ucb = np.dot(feat, param) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update estimates
        feat = bandit.feat(s, a)
        A += np.outer(feat, feat)
        b += feat * reward
        
        pseudoregret = bandit._pseudoregret(s, a)
        cum_pseudoregret += pseudoregret
        regret = bandit._optimal(s) - reward
        cum_regret += regret
        log_row['regret'] = regret
        log_row['pseudoregret'] = pseudoregret
        log_row['cumregret'] = cum_regret
        log_row['cumpseudoregret'] = cum_pseudoregret
        log_row['paramerror'] = np.linalg.norm(param - bandit._param[:dim])
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)
        
    return param

def hoful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, off_scale=1, param_bound=1, off_bound=1, seed=0):    
    logname = 'hoful.%d.%d.%d.%d' % (bandit.n_contexts, bandit.n_arms, bandit.dim, bandit.hidden)
    logger = Logger(directory='../logs', name = logname + '.' + str (seed), modes=['human', 'csv'])
    log_keys = ['pseudoregret',
                 'cumpseudoregret',
                 'regret', 
                 'cumregret', 
                 'paramerror', 
                 'offseterror',
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_pseudoregret = 0
    cum_regret = 0
    
    #Dimensions
    dim = bandit.dim - bandit.hidden
    n = bandit.n_contexts
    m = bandit.n_arms
    
    #Indexing
    def index(s, a):
        return s * m + a
    
    #Magnitude
    S = np.sqrt(param_bound**2 + n*m*off_bound**2 / off_scale**2)
    
    #Initialization
    A = reg * np.eye(dim + n*m)
    #A = np.diag(np.array([reg]*dim + [0.9]*(n*m))) 
    b = np.zeros(dim + n*m)
    
    for t in range(horizon):
        #Latest parameter estimate
        param_and_hid = np.linalg.solve(A, b)
        #param_and_hid = sparse.spsolve(A, b)
        print(np.linalg.det(A))
        
        #Observe context
        s = bandit.observe()
        
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(m):
            feat = bandit.feat(s, i)
            selector = np.zeros(n*m)
            selector[index(s, i)] = off_scale
            X = np.concatenate((feat, selector), axis=0)
            beta = oful_coeff(A, reg, noise, delta, S)
            bonus = beta * inverse_norm(X, A)
            ucb = np.dot(X, param_and_hid) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update estimates
        feat = bandit.feat(s, a) #d
        selector = np.zeros(n*m) #nm
        selector[index(s, a)] = off_scale
        X = np.concatenate((feat, selector), axis=0) #d + nm
        A += np.outer(X, X)
        b += X * reward
        
        param = param_and_hid[:dim]
        hidden_feat = param_and_hid[dim:]
        pseudoregret = bandit._pseudoregret(s, a)
        cum_pseudoregret += pseudoregret
        regret = bandit._optimal(s) - reward
        cum_regret += regret
        est_offset = np.dot(hidden_feat, selector) * off_scale
        log_row['regret'] = regret
        log_row['pseudoregret'] = pseudoregret
        log_row['cumregret'] = cum_regret
        log_row['cumpseudoregret'] = cum_pseudoregret
        log_row['paramerror'] = np.linalg.norm(param - bandit._param[:dim])
        log_row['offseterror'] = np.abs(est_offset - np.dot(bandit._features[s,a,dim:], bandit._param[dim:]))
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)
        
    return param, np.reshape(hidden_feat, (n, m))

def span_hoful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, feat_bound=1, seed=0, fulldim=None):    
    #Dimensions
    dim = bandit.dim - bandit.hidden
    n = bandit.n_contexts
    m = bandit.n_arms
    if fulldim is None:
        fulldim = n*m
    else:
        assert fulldim <= n*m
    
    logname = 'span.%d.%d.%d.%d.%d' % (bandit.n_contexts, bandit.n_arms, bandit.dim, bandit.hidden, fulldim)
    logger = Logger(directory='../logs', name = logname + '.' + str (seed), modes=['human', 'csv'])
    log_keys = ['pseudoregret',
                 'cumpseudoregret',
                 'regret', 
                 'cumregret', 
                 'paramerror', 
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    cum_pseudoregret = 0
    cum_regret = 0
        
    known_basis = feature_basis(bandit.get_features())
    known_rank = known_basis.shape[-1]
    assert fulldim >= known_rank
    hidden_basis = basis_completion(known_basis, fulldim)

    #TODO: introduce concept of full dim
    print(hidden_basis.shape)
    full_features = np.dstack((known_basis, hidden_basis))
    #Normalize
    full_features = full_features / np.linalg.norm(full_features, axis=2, keepdims=True) * feat_bound
    full_param_bound = np.sqrt(n*m) * feat_bound * param_bound

    assert np.linalg.matrix_rank(np.reshape(full_features, (n*m, fulldim))) == full_features.shape[2]
    
    #Initialization
    A = reg * np.eye(fulldim)
    b = np.zeros(fulldim)
    
    for t in range(horizon):
        #Latest parameter estimate
        full_param = np.linalg.solve(A, b)
        
        #Observe context
        s = bandit.observe()
        
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(m):
            feat = full_features[s, i]
            beta = oful_coeff(A, reg, noise, delta, full_param_bound)
            bonus = beta * inverse_norm(feat, A)
            ucb = np.dot(feat, full_param) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update estimates
        feat = full_features[s, a] #d
        A += np.outer(feat, feat)
        b += feat * reward
        
        #Restore original param
        Phi = np.reshape(bandit.get_features(), (n*m, dim))
        olPhi = np.reshape(full_features, (n*m, fulldim))[:, :known_rank]
        param = np.linalg.lstsq(olPhi.T @ Phi, 
                                       full_param[:known_rank], 
                                rcond=None)[0]
        pseudoregret = bandit._pseudoregret(s, a)
        cum_pseudoregret += pseudoregret
        regret = bandit._optimal(s) - reward
        cum_regret += regret
        log_row['regret'] = regret
        log_row['pseudoregret'] = pseudoregret
        log_row['cumregret'] = cum_regret
        log_row['cumpseudoregret'] = cum_pseudoregret
        log_row['paramerror'] = np.linalg.norm(param - bandit._param[:dim])
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        log_row['zeronorm'] = full_param.shape[0] - sum(np.isclose(full_param, 0))
        logger.write_row(log_row, t)
     
    return param, full_param, full_features 

def oful_coeff(A, reg, noise, delta, param_bound):
    dim = A.shape[0]
    return (noise * np.sqrt(2 * np.log(np.sqrt(np.linalg.det(A)) / 
                                              reg**(dim / 2) / delta ))
                        + np.sqrt(reg) * param_bound)