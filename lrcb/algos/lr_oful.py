import numpy as np
from lrcb.utils import inverse_norm
from lrcb.logger import Logger
from lrcb.algos.oful import oful_coeff

def select_oful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, 
         seed=0, verbose=True, logname='oful'):    
    np.random.seed(seed)
    log_modes = ['csv']
    if verbose: 
        log_modes.append('human')
    logger = Logger(directory='../logs', name = logname + '.' + str (seed), modes=log_modes)
    log_keys = ['regret', 
                 'cumregret', 
                 'selection', 
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0
    
    dims = [r.dim for r in bandit.reps]
    A = [reg * np.eye(dim) for dim in dims]
    b = [np.zeros(dim) for dim in dims]
    params = [np.zeros(dim) for dim in dims]
    for t in range(horizon):   
        #Observe context
        s = bandit.observe()
     
        
        #Select representation
        min_eigs = []
        for i, r in enumerate(bandit.reps):
            feats = r.features
            param = np.linalg.solve(A[i], b[i])
            params[i] = param
            rewards = np.matmul(feats, param)
            xx = np.arange(bandit.n_contexts)
            optimal_arms = np.argmax(rewards, axis=1)
            optimal_features = feats[xx, optimal_arms, :]
            _, sv, _ = np.linalg.svd(optimal_features)
            min_eigs.append(sv[-1])
        selection = np.argmax(min_eigs)
        bandit.select_rep(selection)
            
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            feat = bandit.feat(s, i)
            beta = oful_coeff(A[selection], reg, noise, delta, param_bound)
            bonus = beta * inverse_norm(feat, A[selection])
            ucb = np.dot(feat, params[selection]) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update estimates
        for i, r in enumerate(bandit.reps):
            feat = bandit.reps[i].features[s, a, :]
            A[i] += np.outer(feat, feat)
            b[i] += feat * reward
        
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        log_row['selection'] = selection
        logger.write_row(log_row, t)
        
    return param