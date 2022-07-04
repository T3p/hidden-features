from .multiclass import MulticlassToBandit, MCOneHot, MCExpanded
from .linear import LinearContinuous
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from typing import Union


def make_from_dataset(
    name:str, bandit_model:str=None, seed:int=0, 
    noise:str=None, noise_param:float=None,
    rew_optimal:float=1, rew_suboptimal:float=0, shuffle:bool=True):
    # Fetch data
    if name in ['adult_num', 'adult_onehot']:
        X, y = fetch_openml('adult', version=1, return_X_y=True)
        is_NaN = X.isna()
        row_has_NaN = is_NaN.any(axis=1)
        X = X[~row_has_NaN]
        # y = y[~row_has_NaN]
        y = X["occupation"]
        X = X.drop(["occupation"],axis=1)
        cat_ix = X.select_dtypes(include=['category']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        encoder = LabelEncoder()
        # now apply the transformation to all the columns:
        for col in cat_ix:
            X[col] = encoder.fit_transform(X[col])
        y = encoder.fit_transform(y)
        if name == 'adult_onehot':
            cat_features = OneHotEncoder(sparse=False).fit_transform(X[cat_ix])
            num_features = StandardScaler().fit_transform(X[num_ix])
            X = np.concatenate((num_features, cat_features), axis=1)
        else:
            X = StandardScaler().fit_transform(X)
    elif name in ['mushroom_num', 'mushroom_onehot']:
        X, y = fetch_openml('mushroom', version=1, return_X_y=True)
        encoder = LabelEncoder()
        # now apply the transformation to all the columns:
        for col in X.columns:
            X[col] = encoder.fit_transform(X[col])
        # X = X.drop(["veil-type"],axis=1)
        y = encoder.fit_transform(y)
        if name == 'mushroom_onehot':
            X = OneHotEncoder(sparse=False).fit_transform(X)
        else:
            X = StandardScaler().fit_transform(X)
    elif name in ['covertype']:
        # https://www.openml.org/d/150
        # there are some 0/1 features -> consider just numeric
        X, y = fetch_openml('covertype', version=3, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    elif name == 'shuttle':
        # https://www.openml.org/d/40685
        # all numeric, no missing values
        X, y = fetch_openml('shuttle', version=1, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    elif name == 'magic':
        # https://www.openml.org/d/1120
        # all numeric, no missing values
        X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    else:
        raise RuntimeError('Dataset does not exist')

    if bandit_model in [None, "none", "None"]:
        bandit = MulticlassToBandit(X, y, 
        dataset_name=name, seed=seed, noise=noise, noise_param=noise_param,
        rew_optimal=rew_optimal, rew_suboptimal=rew_suboptimal, shuffle=shuffle)
    elif bandit_model == "onehot":
        bandit = MCOneHot(X, y, 
        dataset_name=name, seed=seed, noise=noise, noise_param=noise_param,
        rew_optimal=rew_optimal, rew_suboptimal=rew_suboptimal, shuffle=shuffle)
    elif bandit_model == "expanded":
        bandit = MCExpanded(X, y, 
        dataset_name=name, seed=seed, noise=noise, noise_param=noise_param,
        rew_optimal=rew_optimal, rew_suboptimal=rew_suboptimal, shuffle=shuffle)
    else:
        raise RuntimeError('Bandit model does not exist')
    return bandit


def make_linear(name:str, **kwargs):

    if name == "linear_continuous":
        return LinearContinuous(**kwargs)
    else:
        raise ValueError("unknown name")
    


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
            context = random_problem.randn(n_contexts, dim) * features_sigma + features_mean
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