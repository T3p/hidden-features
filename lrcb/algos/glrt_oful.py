import numpy as np
from lrcb.logger import Logger
from lrcb.algos.oful import oful_coeff_inv
from lrcb.utils import min_eig_outer, sherman_morrison, weighted_norm

def glrt_oful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1,
         robin = True,
         seed=0, verbose=True, logname='glrt_oful'):    
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
                 'reward',
                 'commit',
                 'score']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0
    
    context_counts = np.zeros(bandit.n_contexts)
    
    
    dims = [r.dim for r in bandit.reps]
    invA = [np.eye(dim) / reg for dim in dims]
    b = [np.zeros(dim) for dim in dims]
    params = [np.zeros(dim) for dim in dims]
    commit = None
    for t in range(horizon):   
        #Observe context
        s = bandit.observe()
        
        if commit is None:
            #EXPLORE
            #Update parameters
            for i in range(len(bandit.reps)):
                params[i] = np.matmul(invA[i], b[i])
            
            #Perform GLRT
            for i, r in enumerate(bandit.reps):
                feats = r.features #SxAxd
                rews = np.matmul(feats, params[i]) #SxA
                xx = np.arange(bandit.n_contexts)
                optimal_arms = np.argmax(rews, axis=1) #S
                optimal_feats = feats[xx, optimal_arms, :] #Sxd
                feat_diffs = feats - optimal_feats.reshape((
                    bandit.n_contexts, 1, r.dim)) #SxAxd
                tmp = np.einsum('ijk,kl->ijl', feat_diffs, invA[i]) #SxAxd
                squared_dist = np.einsum('ijk,ijk->ij', feat_diffs, tmp) #SxA
                opt_rews = np.matmul(optimal_feats, params[i]) #S
                gaps = opt_rews.reshape((bandit.n_contexts, 1)) - rews #SxA
                ratios = np.divide(gaps ** 2, squared_dist, 
                                   out=np.inf * np.ones_like(gaps), 
                                   where=~np.isclose(squared_dist, 
                                            np.zeros_like(squared_dist))) #SxA
                min_ratio = np.min(ratios)
                beta = oful_coeff_inv(invA[i], reg, noise, delta, param_bound)**2
                score = min_ratio - beta
                if score > 0:
                    #The test triggers: optimal policy identified
                    #Commit to HLS representation
                    selection = i
                    commit = i
                    bandit.select_rep(commit)
                    #Play optimal action
                    a = optimal_arms[s]
                    reward = bandit.pull(a)
                    break 
            
            if commit is None: #(still)
                #Select representation (round-robin or uniformly at random)
                if robin:
                    selection = t % len(bandit.reps)
                else:
                    selection = np.random.choice(range(len(bandit.reps)))
                bandit.select_rep(selection)
                    
                #Select arm optimistically
                best = -np.inf
                a = 0
                for i in range(bandit.n_arms):
                    feat = bandit.feat(s, i)
                    sqrtbeta = oful_coeff_inv(invA[selection], reg, noise, delta, param_bound)
                    bonus = sqrtbeta * weighted_norm(feat, invA[selection])
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
        
        else:
            #COMMIT
            score = 0
            best = -np.inf
            a = 0
            for i in range(bandit.n_arms):
                feat = bandit.feat(s, i)
                mu = np.dot(feat, params[commit])
                if mu > best:
                    best = mu
                    a = i
            reward = bandit.pull(a)
            
            
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['selection'] = selection
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        log_row['commit'] = 0 if commit is None else 1
        log_row['score'] = score
        logger.write_row(log_row, t)
        
    return params