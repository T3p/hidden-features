import numpy as np
from typing import Union


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = A_inv @ u
    den = 1 + np.dot(u.T, Au)
    A_inv -= np.outer(Au, Au) / (den)
    return A_inv, den

def make_synthetic_features(
    n_contexts: int, n_actions: int, dim: int,
    context_generation: str, feature_expansion: Union[str, None],
    seed: int,
    min_value:float=-1, max_value=1,
    features_sigma=1, features_mean=0, feature_proba=0.5
):
    if feature_expansion in ["none", "None"]:
        feature_expansion = None
    assert context_generation in ["gaussian", "bernoulli", "uniform"]
    assert feature_expansion in [None, "expanded", "onehot"]
    random_problem = np.random.RandomState(seed=seed)
    if feature_expansion is None:
        fdim = dim
        if context_generation == "uniform":
            features = random_problem.uniform(low=min_value, high=max_value, size=(n_contexts, n_actions, dim))
        elif context_generation == "gaussian":
            features = random_problem.randn(n_contexts* n_actions* dim).reshape(n_contexts, n_actions, dim) * features_sigma + features_mean
        elif context_generation == "bernoulli":
            features = random_problem.binomial(1, p=feature_proba, size=(n_contexts, n_actions, dim))
    else:
        if context_generation == "uniform":
            context = random_problem.uniform(low=min_value, high=max_value, size=(n_contexts, dim))
        elif context_generation == "gaussian":
            context = random_problem.randn((n_contexts, dim)) * features_sigma + features_mean
        elif context_generation == "bernoulli":
            context = random_problem.binomial(1, p=feature_proba, size=(n_contexts, dim))

        if feature_expansion == "expanded":
            fdim = dim * n_actions
            features = np.zeros((n_contexts, n_actions, fdim))
            for x in range(n_contexts):
                for a in range(n_actions):
                    features[x, a, a * dim: a * dim + dim] = context[x]
        elif feature_expansion == "onehot":
            fdim = dim + n_actions
            features = np.zeros((n_contexts, n_actions, fdim))
            for x in range(n_contexts):
                for a in range(n_actions):
                    features[x, a, 0:dim] = context[x]
                    features[x, a, a] = 1
    theta = random_problem.uniform(low=min_value, high=max_value, size=fdim)
    return features, theta
