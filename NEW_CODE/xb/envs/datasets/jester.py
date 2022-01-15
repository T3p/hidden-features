import numpy as np
from ..contextualfinite import ContextualFinite


class Jester:
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
        self.user_features = data['user_features'].copy() #dim: nu x du
        self.arm_features = data['arm_features'].copy() #dim: na x da
        self.labels = data['y'].copy() #dim: nu*da x 1
        # self.default_features = data['X'].copy() #dim: nu*da x dim
        self.n_actions = self.arm_features.shape[0]
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        del data

    def reset(self):
        return self.sample_context()

    def get_available_actions(self):
        actions = [{
            "id": a,
            "values": self.arm_features[a]
        } for a in range(self.n_actions)]
        return actions
    
    def sample_context(self):
        idx = self.rng.choice(self.user_features.shape[0], 1).item()
        self.idx = idx
        state = {
            "id": self.idx,
            "values": self.user_features[self.idx]
        }
        return state

    def step(self, action):
        return self.labels[self.idx*self.n_actions+action['id']]

    def best_reward(self):
        return self.labels[self.idx*self.n_actions:(self.idx+1)*self.n_actions].max()
    
    def get_default_representation(self):
        return Jester._rep(self.default_features, self.n_actions)


class Jester_fitted(ContextualFinite):

    def __init__(self, data_file, random_state) -> None:
        data = np.load(data_file)
        X = data['features']
        y = data['predictions']
        super().__init__(features=X, labels=y, random_state=random_state)