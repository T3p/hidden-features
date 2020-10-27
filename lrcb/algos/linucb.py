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

def oful_coeff(A, reg, noise, delta, param_bound):
    dim = A.shape[0]
    return (noise * np.sqrt(2 * np.log(np.sqrt(np.linalg.det(A)) / 
                                              reg**(dim / 2) / delta ))
                        + np.sqrt(reg) * param_bound)