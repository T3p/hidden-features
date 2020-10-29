import numpy as np
from lrcb.utils import inverse_norm, sherman_morrison, weighted_norm
from lrcb.logger import Logger
"""
    Agnostic approach: just use apparent features (misspecified model)
"""
def oful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, 
         sherman=True, seed=0, verbose=True, logname='oful'): 
    if sherman and reg > 0:
        return _oful_sm(bandit, horizon, reg, noise, delta, param_bound,
                             seed, verbose, logname)
    else:
        return _oful_solve(bandit, horizon, reg, noise, delta, param_bound,
                             seed, verbose, logname)

def _oful_solve(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, 
         seed=0, verbose=True, logname='oful'):    
    np.random.seed(seed)
    log_modes = ['csv']
    if verbose: 
        log_modes.append('human')
    logger = Logger(directory='../logs', name = logname + '.' + str (seed), modes=log_modes)
    log_keys = ['regret', 
                 'cumregret', 
                 'paramerror', 
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0
    
    dim = bandit.dim
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
        
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['paramerror'] = np.linalg.norm(param - bandit._param[:dim])
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)
        
    return param

def _oful_sm(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, 
         seed=0, verbose=True, logname='oful'):    
    np.random.seed(seed)
    log_modes = ['csv']
    if verbose: 
        log_modes.append('human')
    logger = Logger(directory='../logs', name = logname + '.' + str (seed), modes=log_modes)
    log_keys = ['regret', 
                 'cumregret', 
                 'paramerror', 
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0
    
    dim = bandit.dim
    invA = np.eye(dim) / reg
    b = np.zeros(dim)
    
    for t in range(horizon):
        #Latest parameter estimate
        param = np.matmul(invA, b)
        
        #Observe context
        s = bandit.observe()
        
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            feat = bandit.feat(s, i)
            beta = oful_coeff_inv(invA, reg, noise, delta, param_bound)
            bonus = beta * weighted_norm(feat, invA)
            ucb = np.dot(feat, param) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update estimates
        feat = bandit.feat(s, a)
        invA = sherman_morrison(invA, feat, feat)
        b += feat * reward
        
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['paramerror'] = np.linalg.norm(param - bandit._param[:dim])
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)
        
    return param

def oful_coeff(A, reg, noise, delta, param_bound):
    dim = A.shape[0]
    return (noise * np.sqrt(2 * np.log(np.sqrt(np.linalg.det(A)) / 
                                              reg**(dim / 2) / delta ))
                        + np.sqrt(reg) * param_bound)

def oful_coeff_inv(invA, reg, noise, delta, param_bound):
    dim = invA.shape[0]
    return (noise * np.sqrt(2 * np.log(1. / (np.sqrt(np.linalg.det(invA)) *
                                              reg**(dim / 2) * delta )))
                        + np.sqrt(reg) * param_bound)