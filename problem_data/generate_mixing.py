import numpy as np
import sys
sys.path.insert(0, '..')
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import fuse_columns
import json

out_file = "mixing_realizable.npy"
seed_problem = 99


json_file = "linrep3.json"
with open(json_file, 'r') as f:
    data = json.load(f)
    theta = np.array(data['param'])
    features = np.array(data['features'])
nc, na, dim = features.shape
print(f"dim: {dim}")
true_reward = features @ theta
problem_gen = np.random.RandomState(seed_problem)
rep_list = []
param_list = []
reference_rep = 0
cols_to_fuse = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0]]
for cols in cols_to_fuse:
    f1, p1 = fuse_columns(features=features, param=theta, cols=cols)
    rep_list.append(f1)
    param_list.append(p1)

print(f"total rep: {len(rep_list)}")

np.save(out_file, np.array([rep_list, param_list, reference_rep], dtype=object), allow_pickle=True)
a,b,c = np.load(out_file, allow_pickle=True)

print(np.all([np.allclose(rep_list[i], a[i]) for i in range(len(rep_list))]))
print(np.all([np.allclose(param_list[i], b[i]) for i in range(len(param_list))]))
print(c == reference_rep)
