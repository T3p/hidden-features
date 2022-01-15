import numpy as np

class Adult:
    class _rep():
        def __init__(self, dim) -> None:
            self.dim = dim
        
        def features_dim(self):
            return self.dim
        
        def get_features(self, context, action):
            return np.concatenate([context['values'], action['values']])

    def __init__(self, data_file, random_state=0) -> None:
        self.data_file = data_file,
        data = np.load(data_file)
        self.contexts = data['contexts'].copy() #dim: nc x d
        classes = data['classes'].copy() #dim: nc x 1
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        del data
        
        act_names, self.classes_idx = np.unique(classes, return_inverse=True)
        self.n_actions = len(act_names)
        self.action_features = np.eye(self.n_actions)

    def reset(self):
        return self.sample_context()

    def get_available_actions(self):
        actions = [{
            "id": a,
            "values": self.action_features[a]
        } for a in range(self.n_actions)]
        return actions
    
    def sample_context(self):
        idx = self.rng.choice(self.contexts.shape[0], 1).item()
        self.idx = idx
        state = {
            "id": self.idx,
            "values": self.contexts[self.idx]
        }
        return state

    def step(self, action):
        class_loss = 1 if action["id"] == self.classes_idx[self.idx] else 0
        return class_loss

    def best_reward(self):
        return 1
    
    def get_default_representation(self):
        return Adult._rep(self.default_features, self.n_actions)