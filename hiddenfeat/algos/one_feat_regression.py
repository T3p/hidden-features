import numpy as np
import time
import cvxpy as cp

def solve(bandit, d=1, lambda1=0.1, lambda2=0.1, samples=100):
    assert bandit.hidden == 1
    bot = time.time()
    dim = bandit.dim - bandit.hidden
    m = bandit.n_contexts * bandit.n_arms
    
    def index(s, a):
        return s * bandit.n_arms + a
    
    def de_index(k):
        return k // bandit.n_arms, k % bandit.n_arms

    param = np.zeros(dim)
    hf = np.zeros(m)

    count = np.zeros(m)
    A = lambda1 * np.eye(dim)    
    alpha = np.zeros(m)
    beta = np.zeros((m, m))
    gamma = np.zeros(m)
    sumr = np.zeros(m)
    sumfr = np.zeros(dim)
    indexes = []
    
    param_errors = []
    pred_errors = []
    
    for t in range(samples):
        start = time.time()
        #Perform random action
        s = bandit.observe()
        a = np.random.choice(bandit.n_arms)
        indexes.append(index(s, a))
        
        pred = np.dot(param, bandit.feat(s,a)) + hf[index(s,a)] * d
        r = bandit.pull(a)
        pred_errors.append(bandit._expected_reward(s, a) - pred)
        
        feat = bandit.feat(s, a)
        count[index(s, a)] += 1
        sumr[index(s, a)] += r
        sumfr += feat * r
        
        #Design matrix
        A += np.outer(feat, feat)
        
        #Estimate hidden feature
        for k in range(m):
            featk = bandit.feat(*de_index(k))
            alpha[k] = lambda2 + count[k] * d**2 * (1 + count[k] * 
                         np.dot(featk, np.linalg.solve(A, featk)))
            for j in range(m):
                featj = bandit.feat(*de_index(j))
                beta[k, j] = count[k] * count[j]* d**2 * np.dot(featk, 
                                np.linalg.solve(A, featj))
            
            gamma[k] = d * (sumr[k] - count[k] * np.dot(featk, np.linalg.solve(A, sumfr)))
            
        M = beta - np.diag(np.diag(beta)) + np.diag(alpha)
        hf = np.linalg.solve(M, gamma)
        
        #Estimate parameter
        b = sumfr
        for i in range(t):
            b -= d * bandit.feat(*de_index(indexes[i])) * hf[indexes[i]]
        param = np.linalg.solve(A, b)
        
        print(param)
        
        param_errors.append(np.linalg.norm(param - bandit._param[:-1]))
        print('%d / %d (%f s)' % (t, samples, time.time() - start))
    
    print('Total time: %f s' % (time.time() - bot))
    return param, hf, param_errors, pred_errors

def solve_auto(bandit, d=1, lambda1=0.1, lambda2=0.1, samples=100):
    assert bandit.hidden == 1
    bot = time.time()
    dim = bandit.dim - bandit.hidden
    m = bandit.n_contexts * bandit.n_arms
    
    def index(s, a):
        return s * bandit.n_arms + a
    
    def de_index(k):
        return k // bandit.n_arms, k % bandit.n_arms

    param = cp.Variable(dim)
    param.value = np.zeros(dim)
    hf = cp.Variable(m)
    hf.value = np.zeros(m)
    
    rews = []
    feat_mat = None #t X d
    D = None #t x m
    
    param_errors = []
    pred_errors = []
    
    for t in range(samples):
        start = time.time()
        #Perform random action
        s = bandit.observe()
        a = np.random.choice(bandit.n_arms)
        feat = bandit.feat(s, a)
        
        if feat_mat is None:
            feat_mat = feat.reshape((1, dim))
        else:
            feat_mat = np.concatenate((feat_mat, feat.reshape((1, dim))), axis=0)
        if D is None:
            D = np.zeros((1,m))
            D[0, index(s,a)] = d
        else:
            drow = np.zeros((1,m))
            drow[0, index(s,a)] = d
            D = np.concatenate((D, drow), axis=0) 
        
        hfv = hf.value[index(s,a)] if hf.value is not None else 0
        paramv = param.value if param.value is not None else np.zeros(dim)
        pred = np.dot(paramv, bandit.feat(s,a)) + hfv * d
        r = bandit.pull(a)
        rews.append(r)
        pred_errors.append(bandit._expected_reward(s, a) - pred)
        
        #Solve using cvxpy
        cost = cp.sum_squares(feat_mat @ param + D @ hf - np.array(rews)) + lambda1 * cp.sum_squares(param) + lambda2 * cp.sum_squares(hf)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()    
        
        print(param.value)

        param_errors.append(np.linalg.norm(param.value - bandit._param[:-1]))
        print('%d / %d (%f s)' % (t, samples, time.time() - start))
    
    print('Total time: %f s' % (time.time() - bot))
    return param, hf, param_errors, pred_errors
    