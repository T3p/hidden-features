import numpy as np
from lrcb.logger import Logger
from lrcb.algos.oful import oful_coeff_inv
from lrcb.utils import min_eig_outer, sherman_morrison, weighted_norm

def select_oful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1,
                rule='maxlambdamin',
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
    invA = [np.eye(dim) / reg for dim in dims]
    b = [np.zeros(dim) for dim in dims]
    params = [np.zeros(dim) for dim in dims]
    for t in range(horizon):   
        #Observe context
        s = bandit.observe()
     
        #Update parameters
        for i in range(len(bandit.reps)):
            params[i] = np.matmul(invA[i], b[i])
        
        #Select representation
        selection = 0
        if rule=='maxlambdamin':
            maxmin_eig = 0
            for i, r in enumerate(bandit.reps):
                feats = r.features
                rewards = np.matmul(feats, params[i])
                xx = np.arange(bandit.n_contexts)
                optimal_arms = np.argmax(rewards, axis=1)
                optimal_features = feats[xx, optimal_arms, :]
                min_eig = min_eig_outer(optimal_features)
                if min_eig > maxmin_eig:
                    maxmin_eig = min_eig
                    selection = i
        elif rule=='minbonus':
            minbestbonus = np.inf
            selection = 0
            for i, r in enumerate(bandit.reps):
                feats = r.features
                best = -np.inf
                bestbonus = 0
                for k in range(bandit.n_arms):
                    feat = feats[s, k]
                    beta = oful_coeff_inv(invA[i], reg, noise, delta, param_bound)
                    bonus = beta * weighted_norm(feat, invA[i])
                    ucb = np.dot(feat, params[i]) + bonus
                    if ucb > best:
                        best = ucb
                        bestbonus = bonus
                if bestbonus < minbestbonus:
                    minbestbonus = bestbonus
                    selection = i
        else:
            raise NotImplementedError()
            
        bandit.select_rep(selection)
            
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            feat = bandit.feat(s, i)
            beta = oful_coeff_inv(invA[selection], reg, noise, delta, param_bound)
            bonus = beta * weighted_norm(feat, invA[selection])
            ucb = np.dot(feat, params[selection]) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update estimates
        for i, r in enumerate(bandit.reps):
            feat = bandit.reps[i].features[s, a, :]
            invA[i] = sherman_morrison(invA[i], feat, feat)
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
        
    return params