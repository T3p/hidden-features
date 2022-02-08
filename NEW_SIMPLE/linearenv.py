import numpy as np

class LinearEnv():
    def __init__(self, features: np.array, param: np.array, rew_noise: float=0.5, random_state: int=0) -> None:
        self.features = features
        self.param = param
        self.rewards = features @ param
        self.rew_noise = rew_noise
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.n_contexts, self.n_actions, self.feat_dim = self.features.shape

    def get_available_actions(self) -> np.ndarray:
        """ Return the actions available at each time
            Return: n_actions x dim_action_features
        """
        actions = np.arange(self.n_actions)
        return actions
    
    def sample_context(self) -> np.ndarray:
        """ Return a random context
        """
        self.idx = self.rng.choice(self.n_contexts, 1)
        return self.idx

    def step(self, action: int) -> float:
        """ Return a realization of the reward in the context for the selected action
        """
        return self.rewards[self.idx, action] + self.rng.randn() * self.rew_noise

    def best_reward(self) -> float:
        """ Maximum reward in the current context
        """
        return self.rewards[self.idx].max()
    
    def expected_reward(self, action: int) -> float:
        return self.rewards[self.idx, action]

class LinearRepresentation():
    """ Returns the features associated to each context and action
    """
    def __init__(self, features: np.ndarray) -> None:
        self.features = features
    
    def features_dim(self) -> int:
        return self.features.shape[2]
    
    def get_features(self, context: np.ndarray, action: int) -> np.ndarray:
        return self.features[context[0], action]
