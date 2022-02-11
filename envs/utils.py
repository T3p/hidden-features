from .multiclass import MulticlassToBandit, MCOneHot, MCExpanded
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def make_from_dataset(name:str, bandit_model:str=None, seed:int=0, noise:str=None, noise_param:float=None):
    # Fetch data
    if name in ['adult_num', 'adult_onehot']:
        X, y = fetch_openml('adult', version=1, return_X_y=True)
        X = X.dropna()
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
    elif name in ['mushroom_num','mushroom_onehot']:
        X, y = fetch_openml('mushroom', version=1, return_X_y=True)
        encoder = LabelEncoder()
        # now apply the transformation to all the columns:
        for col in X.columns:
            X[col] = encoder.fit_transform(X[col])
        X = X.drop(["veil-type"],axis=1)
        y = encoder.fit_transform(y)
        if name == 'mushroom_onehot':
            X = OneHotEncoder(sparse=False).fit_transform(X)
        else:
            X = StandardScaler().fit_transform(X)
    else:
        raise RuntimeError('Dataset does not exist')

    if bandit_model is None:
        bandit = MulticlassToBandit(X, y, dataset_name=name, seed=seed, noise=noise, noise_param=noise_param)
    elif bandit_model == "onehot":
        bandit = MCOneHot(X, y, dataset_name=name, seed=seed, noise=noise, noise_param=noise_param)
    elif bandit_model == "expanded":
        bandit = MCExpanded(X, y, dataset_name=name, seed=seed, noise=noise, noise_param=noise_param)
    else:
        raise RuntimeError('Bandit model does not exist')
    return bandit
