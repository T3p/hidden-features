import numpy as np
from lrcb.logger import Logger
from lrcb.algos.oful import oful_coeff_inv
from lrcb.utils import min_eig_outer, sherman_morrison, weighted_norm
import numbers

def maximin_oful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bounds=1.,
         seed=0, verbose=True, logname='maximin'):    
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
    
    if isinstance(param_bounds, numbers.Number):
        param_bounds = [1.] * len(bandit.reps)
    
    dims = [r.dim for r in bandit.reps]
    invA = [np.eye(dim) / reg for dim in dims]
    b = [np.zeros(dim) for dim in dims]
    params = [np.zeros(dim) for dim in dims]
    for t in range(horizon):   
        #Observe context
        s = bandit.observe()
     
        #Update parameters
        for j in range(len(bandit.reps)):
            params[j] = np.matmul(invA[j], b[j])

        #Select arm and representation
        maxminucb = -np.inf
        for i in range(bandit.n_arms):
            minucb = np.inf
            for j, r in enumerate(bandit.reps):
                feat = r.features[s, i]
                sqrtbeta = oful_coeff_inv(invA[j], reg, noise, delta, param_bounds[j])
                bonus = sqrtbeta * weighted_norm(feat, invA[j])
                ucb = np.dot(feat, params[j]) + bonus
                if ucb < minucb:
                    minucb = ucb
                    argminucb = j
            if minucb > maxminucb:
                maxminucb = minucb
                a = i
                selection = argminucb
        
        bandit.select_rep(selection)
        reward = bandit.pull(a)
        
        #Update estimates
        for j, r in enumerate(bandit.reps):
            feat = bandit.reps[j].features[s, a, :]
            invA[j] = sherman_morrison(invA[j], feat, feat)
            b[j] += feat * reward
        
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        log_row['selection'] = selection
        logger.write_row(log_row, t)
        
    return params