import numpy as np

class ContextualFinite():

    class _rep():
        def __init__(self, features) -> None:
            self.features = features
        
        def features_dim(self):
            return self.features.shape[2]
        
        def get_features(self, context, action):
            return self.features[context['id'], action['id']]
        
        def feature_bound(self):
            return np.amax(np.linalg.norm(self.features, axis=-1))

    def __init__(self, features, labels, random_state=0, noise_std=0.3) -> None:
        self.features = features
        self.labels = labels
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.noise_std = noise_std

        self.n_contexts, self.n_actions, self.feat_dim = self.features.shape

    def reset(self):
        return self.sample_context()
    
    def get_available_actions(self):
        actions = [{
            "id": a,
            "values": None
        } for a in range(self.n_actions)]
        return actions
    
    def sample_context(self):
        idx = self.rng.choice(self.n_contexts, 1).item()
        self.idx = idx
        state = {
            "id": self.idx,
            "values": None
        }
        return state

    def step(self, action):
        return self.labels[self.idx, action['id']] + self.rng.randn() * self.noise_std

    def best_reward(self):
        return self.labels[self.idx].max()
    
    def expected_reward(self, action):
        return self.labels[self.idx][action["id"]]

    def get_default_representation(self):
        return ContextualFinite._rep(self.features)
