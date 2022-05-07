import numpy as np

class LinearEnv():
    def __init__(self, features, param, rew_noise=0.5, random_state=0) -> None:
        self.features = features
        self.param = param
        self.rewards = features @ param
        self.rew_noise = rew_noise
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.n_contexts, self.n_actions, self.feat_dim = self.features.shape

    def get_available_actions(self):
        """ Return the actions available at each time
        """
        actions = np.arange(self.n_actions)
        return actions
    
    def sample_context(self):
        """ Return a random context
        """
        self.idx = self.rng.choice(self.n_contexts, 1).item()
        return self.idx

    def step(self, action):
        """ Return a realization of the reward in the context for the selected action
        """
        return self.rewards[self.idx, action] + self.rng.randn() * self.rew_noise

    def best_reward(self):
        """ Maximum reward in the current context
        """
        return self.rewards[self.idx].max()
    
    def expected_reward(self, action):
        return self.rewards[self.idx, action]
    
    def min_suboptimality_gap(self, tol=1e-6):
        min_gap = np.inf
        for ctx in range(self.n_contexts):
            arr = sorted(self.rewards[ctx])
            for i in range(self.n_actions-1):
                diff = arr[self.n_actions-1] - arr[i]
                if diff < min_gap - tol and diff > tol:
                    min_gap = diff
        return min_gap

class LinearRepresentation():
    """ Returns the features associated to each context and action
    """
    def __init__(self, features) -> None:
        self.features = features
    
    def features_dim(self):
        return self.features.shape[2]
    
    def get_features(self, context, action):
        return self.features[context, action]