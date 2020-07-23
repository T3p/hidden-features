from hiddenfeat.utils import feature_basis, get_orthogonal, complete_basis
from hiddenfeat.bandits.finite_linear_bandits import make_random_hflb
import numpy as np

bandit = make_random_hflb(n_contexts=2, n_arms=2, dim=4, hidden=1, orthogonalize=False)

f = bandit.get_features()
print(f, '\n------------------\n')

ff = complete_basis(f)
print(ff)