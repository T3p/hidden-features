import numpy as np
from hiddenfeat.utils import inverse_norm
from hiddenfeat.logger import Logger

"""
    Agnostic approach: just use apparent features (misspecified model)
"""
def oful(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, seed=0):    
    logname = 'oracle' if bandit.hidden == 0 else 'oful'
    logger = Logger(directory='../logs', name = logname + '.' + str (seed), modes=['human', 'csv'])
    log_keys = ['regret', 
                 'cumregret', 
                 'paramerror', 
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
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

def hoful(bandit, horizon, reg1=0.1, reg2=0.1, noise=0.1, delta=0.1, param_bound=1, feat_bound=1, seed=0):
    """
    Variant of Wang et al. 2016, non-incremental version
    """
    logger = Logger(directory='../logs', name = 'hoful.%d' % seed, modes=['human', 'csv'])
    log_keys = ['regret', 
             'cumregret', 
             'paramerror', 
             'hparamerror', 
             'kparamerror', 
             'context', 
             'arm', 
             'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0

    dim = bandit.dim
    hdim = bandit.hidden
    kdim = dim - hdim
    A = reg1 * np.eye(dim)
    b = np.random.uniform(0, 1, size=dim)
    C = reg2 * np.eye(hdim)
    C = np.tile(C, (bandit.n_contexts, bandit.n_arms, 1, 1))
    d = np.zeros(hdim)
    d = np.tile(d, (bandit.n_contexts, bandit.n_arms, 1))
    counts = np.zeros((bandit.n_contexts, bandit.n_arms))
    contexts = []
    arms = []
    rewards = []
    rew_sums = np.zeros((bandit.n_contexts, bandit.n_arms))
    hfeats = np.zeros((bandit.n_contexts, bandit.n_arms, hdim))
    param = np.zeros(dim)
    kparam = param[:kdim]
    hparam = param[kdim:]
    
    for t in range(horizon):
        #Observe context
        s = bandit.observe()
    
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            feat = np.concatenate((bandit.feat(s, i), hfeats[s, i]))
            beta = oful_coeff(A, reg1, noise, delta, param_bound)
            alpha = oful_coeff(C[s, i], reg2, noise, delta, feat_bound)
            bonus = beta * inverse_norm(feat, A) + alpha * inverse_norm(hparam, 
                                       C[s, i])
            ucb = np.dot(feat, param) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update parameter estimate
        X = np.concatenate((bandit.feat(s, a), hfeats[s, a]))
        A += np.outer(X, X)
        b += X * reward
        param = np.linalg.solve(A, b)
        if np.linalg.norm(param) > param_bound:
            param *= param_bound / np.linalg.norm(param)
        kparam = param[:kdim]
        hparam = param[kdim:]
        
        #Update memory
        contexts.append(s)
        arms.append(a)
        counts[s, a] += 1
        rewards.append(reward)
        rew_sums[s, a] += reward
        
        #Revise previous estimates
        A = reg1 * np.eye(dim) 
        b = 0
        for sk, ak, rk in zip(contexts, arms, rewards):
            tk = counts[sk, ak]
            C[sk, ak] = reg2 * np.eye(hdim) + tk * np.outer(hparam, hparam)
            featk = bandit.feat(sk, ak)
            d[sk, ak] = hparam * (rew_sums[sk, ak] - tk * np.dot(featk, kparam))
            hfeat = np.linalg.solve(C[sk, ak], d[sk, ak])
            Xk = np.concatenate((bandit.feat(sk, ak), hfeat))
            if np.linalg.norm(Xk) > feat_bound:
                hfeat *= (np.sqrt(feat_bound**2 - np.linalg.norm(featk)**2) /
                                np.linalg.norm(hfeat))
            hfeats[sk, ak] = hfeat
            Xk = np.concatenate((bandit.feat(sk, ak), hfeat))
            A += np.outer(Xk, Xk)
            b += Xk * rk

        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['paramerror'] = np.linalg.norm(param - bandit._param)
        log_row['hparamerror'] = np.linalg.norm(hparam - bandit._param[kdim:])
        log_row['kparamerror'] = np.linalg.norm(kparam - bandit._param[:kdim])
        log_row['featerror'] = np.linalg.norm(hfeats - bandit._features[:,:,kdim:])
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)
        
    return param, hfeats

def incr_hoful(bandit, horizon, reg1=0.1, reg2=0.1, noise=0.1, delta=0.1, param_bound=1, feat_bound=1, seed=0):
    """
    Variant of Wang et al. 2016, incremental version
    """
    logger = Logger(directory='../logs', name = 'ihoful.%d' % seed, modes=['human', 'csv'])
    log_keys = ['regret', 
                 'cumregret', 
                 'paramerror', 
                 'hparamerror', 
                 'kparamerror', 
                 'featerror',
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0
    
    dim = bandit.dim
    hdim = bandit.hidden
    kdim = dim - hdim
    A = reg1 * np.eye(dim)
    b = np.random.uniform(0, 1, size=dim)
    C = reg2 * np.eye(hdim)
    C = np.tile(C, (bandit.n_contexts, bandit.n_arms, 1, 1))
    d = np.zeros(hdim)
    d = np.tile(d, (bandit.n_contexts, bandit.n_arms, 1))
    hfeats = np.zeros((bandit.n_contexts, bandit.n_arms, hdim))
    param = np.zeros(dim)
    kparam = param[:kdim]
    hparam = param[kdim:]
    
    for t in range(horizon):
        #Observe context
        s = bandit.observe()
    
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            feat = np.concatenate((bandit.feat(s, i), hfeats[s, i]))
            beta = oful_coeff(A, reg1, noise, delta, param_bound)
            alpha = oful_coeff(C[s, i], reg2, noise, delta, feat_bound)
            bonus = beta * inverse_norm(feat, A) + alpha * inverse_norm(hparam, 
                                       C[s, i])
            ucb = np.dot(feat, param) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update parameter estimate
        X = np.concatenate((bandit.feat(s, a), hfeats[s, a]))
        A += np.outer(X, X)
        b += X * reward
        param = np.linalg.solve(A, b)
        if np.linalg.norm(param) > param_bound:
            param *= param_bound / np.linalg.norm(param)
        kparam = param[:kdim]
        hparam = param[kdim:]
        
        C[s, a] += np.outer(hparam, hparam)
        featk = bandit.feat(s, a)
        d[s, a] += hparam * (reward - np.dot(featk, kparam))
        hfeat = np.linalg.solve(C[s, a], d[s, a])
        hfeats[s, a] = hfeat
        if np.linalg.norm(hfeat) > feat_bound:
            hfeat *= (np.sqrt(feat_bound**2 - np.linalg.norm(featk)**2) /
                                np.linalg.norm(feat))
            hfeats[s, a] = hfeat
        
        #Keep trace of regret
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['paramerror'] = np.linalg.norm(param - bandit._param)
        log_row['hparamerror'] = np.linalg.norm(hparam - bandit._param[kdim:])
        log_row['kparamerror'] = np.linalg.norm(kparam - bandit._param[:kdim])
        log_row['featerror'] = np.linalg.norm(hfeats - bandit._features[:,:,kdim:])
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)

    return param, hfeats

def offset_approach(bandit, horizon, reg=0.1, noise=0.1, delta=0.1, param_bound=1, feat_bound=1, seed=0):
    """
    Model unknown part as context-arm-dependent offset
    """
    logger = Logger(directory='../logs', name = 'offset.%d' % seed, modes=['human', 'csv'])
    log_keys = ['regret', 
                 'cumregret', 
                 'kparamerror',
                 'offseterror',
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0
    
    dim = bandit.dim - bandit.hidden
    A = reg * np.eye(dim)
    b = np.random.uniform(0, 1, size=dim)
    param = np.zeros(dim)
    offset = np.zeros((bandit.n_contexts, bandit.n_arms))
    counts = np.zeros((bandit.n_contexts, bandit.n_arms))

    for t in range(horizon):
        #Observe context
        s = bandit.observe()
    
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            feat = bandit.feat(s, i)
            beta = oful_coeff(A, reg, noise, delta, param_bound)
            offset_ucb = (offset[s, i] + np.sqrt(2 * np.log(t) / counts[s, i])
                            if counts[s, i] > 0 else np.inf)
            bonus = beta * inverse_norm(feat, A) + offset_ucb
            ucb = np.dot(feat, param) + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update parameter estimate
        X = bandit.feat(s, a)
        A += np.outer(X, X)
        b += X * (reward - offset[s, a])
        param = np.linalg.solve(A, b)
        
        #Update offset estimate
        counts[s, a] += 1
        offset[s, a] = 1 / counts[s, a] * ((counts[s, a] - 1) * offset[s, a] +
                          reward - np.dot(param, X))
        
        #Keep trace of regret
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['kparamerror'] = np.linalg.norm(param - bandit._param[:dim])
        log_row['offseterror'] = np.linalg.norm(offset - np.dot(
                bandit._features[:,:,dim:], bandit._param[dim:]))
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)

    return param, offset

def ucb1(bandit, horizon, seed=0):
    """
    Ignore structure
    """
    logger = Logger(directory='../logs', name = 'ucb.%d' % seed, modes=['human', 'csv'])
    log_keys = ['regret', 
                 'cumregret', 
                 'offseterror',
                 'context', 
                 'arm', 
                 'reward']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    cum_regret = 0
    
    mu = np.zeros((bandit.n_contexts, bandit.n_arms))
    counts = np.zeros((bandit.n_contexts, bandit.n_arms))

    for t in range(horizon):
        #Observe context
        s = bandit.observe()
    
        #Select arm optimistically
        best = -np.inf
        a = 0
        for i in range(bandit.n_arms):
            bonus = (np.sqrt(2 * np.log(t) / counts[s, i])
                            if counts[s, i] > 0 else np.inf)
            ucb = mu[s, i] + bonus
            if ucb > best:
                best = ucb
                a = i
        reward = bandit.pull(a)
        
        #Update offset estimate
        counts[s, a] += 1
        mu[s, a] = 1 / counts[s, a] * ((counts[s, a] - 1) * mu[s, a] + reward)
        
        #Keep trace of regret
        regret = bandit._regret(s, a)
        cum_regret += regret
        log_row['regret'] = regret
        log_row['cumregret'] = cum_regret
        log_row['offseterror'] = np.linalg.norm(mu - np.dot(
                bandit._features, bandit._param))
        log_row['context'] = s
        log_row['arm'] = a
        log_row['reward'] = reward
        logger.write_row(log_row, t)

    return mu

def oful_coeff(A, reg, noise, delta, param_bound):
    dim = A.shape[0]
    return (noise * np.sqrt(2 * np.log(np.sqrt(np.linalg.det(A)) / 
                                              reg**(dim / 2) / delta ))
                        + np.sqrt(reg) * param_bound)