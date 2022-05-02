import numpy as np
import sys
sys.path.insert(0, '..')
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import derank_hls, hls_lambda, is_hls, reduce_dim

in_features_file = "basic_features.npy"
in_theta_file = "basic_param.npy"
seed_problem = 99

out_file = "vardimtest_c1.npy"

features = np.load(in_features_file)
theta = np.load(in_theta_file)
dim = len(theta)
true_reward = features @ theta
problem_gen = np.random.RandomState(seed_problem)

rep_list = []
param_list = []
rep_list.append(features)
param_list.append(theta)
for i in range(2, dim+1):
    fi, pi = reduce_dim(features=features, param=theta, newdim=i, transform=True, normalize=True, seed=seed_problem)
    fi, pi = derank_hls(features=fi, param=pi, newrank=1, transform=True, normalize=True, seed=seed_problem)
    rep_list.append(fi)
    param_list.append(pi)

reference_rep = 0

np.save(out_file, np.array([rep_list, param_list, reference_rep], dtype=object), allow_pickle=True)
a,b,c = np.load(out_file, allow_pickle=True)

print(np.all([np.allclose(rep_list[i], a[i]) for i in range(len(rep_list))]))
print(np.all([np.allclose(param_list[i], b[i]) for i in range(len(param_list))]))
print(c == reference_rep)
