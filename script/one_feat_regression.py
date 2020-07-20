import numpy as np

def solve(bandit, d=1, lambda1=0.1, lambda2=0.1, samples=100):
    assert bandit.hidden == 1
    dim = bandit.dim - bandit.hidden
    m = bandit.n_contexts * bandit.n_actions
    
    def index(s, a):
        return s * bandit.n_actions + a
    
    def de_index(k):
        return k // bandit.n_actions, k % bandit.n_actions

    param = np.zeros(dim)
    hf = np.zeros(m)

    count = np.zeros(m)
    A = lambda1 * np.eye((dim, dim))    
    alpha = np.zeros(m)
    beta = np.zeros((m, m))
    gamma = np.zeros(m)
    sumr = np.zeros(m)
    sumfr = np.zeros(dim)
    
    errors = []
    
    for t in range(samples):
        #Perform random action
        s = bandit.observe()
        a = np.random.choice(bandit.n_arms)
        r = bandit.pull(a)
        feat = bandit.feat(s, a)
        count[index(s, a)] += 1
        sumr[index(s, a)] += r
        sumfr += feat * r
        
        #Design matrix
        A += np.outer(feat, feat)
        
        #Estimate hidden feature
        for k in range(m):
            featk = bandit.feat[de_index(k)]
            alpha[k] = lambda2 + count[k] * d**2 * (1 + count[k] * 
                         np.dot(featk, np.linalg.solve(A, featk)))
            for j in range(m):
                featj = bandit.feat[de_index(j)]
                beta[k, j] = count[k] * count[j]* d**2 * np.dot(featk, 
                                np.linalg.solve(A, featj))
            
            gamma[k] = d * (sumr[k] - count[k] * np.dot(featk, np.linalg.solve(A, sumfr)))
            
        M = beta - np.diag(np.diag(beta)) + np.diag(alpha)
        hf = np.linalg.solve(M, gamma)
        
        #Estimate parameter
        b = sumfr - d * np.sum(count * hf)
        param = np.linalg.solve(A, b)
        
        errors.append[np.linalg.norm(param - bandit._param[:-1])]
    
    return param, hf, errors
    