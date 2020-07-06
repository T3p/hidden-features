import numpy as np
from sklearn.decomposition import PCA

class FiniteLinearBandit:
    def __init__(self, n_contexts, n_arms, features, param, noise=0.1):
        """
            features: n_contexts X n_arms X len(param)
        """
        self.dim = len(param)
        assert (features.shape[0] == n_contexts and features.shape[1] == n_arms 
                and features.shape[2] == self.dim)
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self._features = features
        self._param = param
        self._noise = noise
        
        self._context = None
        
    def observe(self):
        self._context = np.random.choice(self.n_contexts)
        return self._context
    
    def pull(self, arm):
        if self._context is None:
            raise ValueError('Observe context first')
        
        feat = self._features[self._context, arm]
        reward = np.dot(self._param, feat)
        
        self._context = None
        return reward + np.random.normal(0, self._noise)
    
    def feat(self, context, arm):
        return self._features[context, arm]
    
    def _regret(self, context, arm):
        feats = self._features[context]
        values = np.dot(feats, self._param)
        best = np.amax(values)
        return best - np.dot(feats[arm], self._param)
    
class HiddenFiniteLinearBandit(FiniteLinearBandit):
    def __init__(self, n_contexts, n_arms, features, param, noise=0.1, hidden=0):
        super(HiddenFiniteLinearBandit, self).__init__(n_contexts, n_arms,
                 features, param, noise)
        self.hidden = hidden
        
    def feat(self, context, arm):
        if self.hidden > 0: 
            return self._features[context, arm, :-self.hidden]
        return self._features[context, arm]

def make_random_hflb(n_contexts, n_arms, dim, noise=0.1, hidden=0):
    #Generate features
    features = np.zeros((n_contexts, n_arms, dim))
    for i in range(n_contexts):
        for j in range(n_arms):
            var = np.random.uniform(0, 1, size=dim)
            features[i, j] = np.random.normal(np.zeros(dim), np.sqrt(var))
    
    #Orthogonalize features
    features = np.reshape(features, (n_contexts * n_arms, dim))
    orthogonalizer = PCA(n_components=dim) #no dimensionality reduction
    features = orthogonalizer.fit_transform(features)
    features = np.reshape(features, (n_contexts, n_arms, dim))

    #Normalize
    for i in range(n_contexts):
        for j in range(n_arms):
            features[i, j] /= np.linalg.norm(features[i, j])
    
    #Generate params
    param = np.random.uniform(0, 1, size=dim)
    #Normalize
    param /= np.linalg.norm(param)
    
    return HiddenFiniteLinearBandit(n_contexts, n_arms, features, param, noise, 
                                    hidden)