import numpy as np
import sys
sys.path.insert(0, '..')
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import derank_hls, hls_lambda, hls_rank, is_hls, reduce_dim, make_reshaped_linrep, rank
import json
from lbrl.utils import make_synthetic_features

out_file = "vardimtest_icml_nonrealizable.npy"
# out_file = "vardimtest_icml_realizable.npy"
# out_file = "vardimtest_icml_real_nohls.npy"
out_file = "vardim_weakhls.npy"
seed_problem = 99

# in_features_file = "basic_features.npy"
# in_theta_file = "basic_param.npy"
# features = np.load(in_features_file)
# theta = np.load(in_theta_file)

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



if out_file in ["vardimtest_icml_realizable.npy", "vardim_weakhls.npy"]:
    rep_list.append(features)
    param_list.append(theta)
for i in range(2, dim+1):
    fi, pi = reduce_dim(features=features, param=theta, newdim=i, transform=True, normalize=True, seed=seed_problem)
    fi, pi = derank_hls(features=fi, param=pi, newrank=1, transform=True, normalize=True, seed=seed_problem)
    rep_list.append(fi)
    param_list.append(pi)


if out_file == "vardim_weakhls.npy":
    new_rep_list = []
    new_parm_list = []
    for i in range(len(rep_list)):
        fi, pi = rep_list[i], param_list[i]
        const_feat = np.ones((fi.shape[0], fi.shape[1], 5))
        fnew = np.concatenate((fi, const_feat), axis=-1)
        pnew = np.concatenate((pi, np.zeros(5)))
        new_rep_list.append(fnew)
        new_parm_list.append(pnew)

        print(f"dim: {pnew.shape}")
        print(f"hls_rank: {hls_rank(fnew, fnew @ pnew)}")
        print(f"tot rank: {rank(fnew)}")

reference_rep = 0
if out_file == "vardimtest_icml_nonrealizable.npy":
    f1, p1 = make_reshaped_linrep(features=features, param=theta, newdim=int(dim/2), transform=True, normalize=True, seed=seed_problem)
    rep_list.append(f1)
    param_list.append(p1)
    print(f1.shape, p1.shape)
    f2, p2 = make_reshaped_linrep(features=features, param=theta, newdim=int(dim/3), transform=True, normalize=True, seed=seed_problem)
    rep_list.append(f2)
    param_list.append(p2)
    print(f2.shape, p2.shape)
    f3, p3 = make_synthetic_features(n_contexts=nc, n_actions=na, dim=3,
        context_generation="uniform", feature_expansion="none", seed=seed_problem)
    rep_list.append(f3)
    param_list.append(p3)
    print(f3.shape, p3.shape)
    f4, p4 = make_synthetic_features(n_contexts=nc, n_actions=na, dim=9,
        context_generation="uniform", feature_expansion="none", seed=10*seed_problem)
    print(f4.shape, p4.shape)
    rep_list.append(f4)
    param_list.append(p4)
    f4, p4 = make_synthetic_features(n_contexts=nc, n_actions=na, dim=18,
        context_generation="bernoulli", feature_expansion="none", seed=seed_problem)
    print(f4.shape, p4.shape)
    rep_list.append(f4*5)
    param_list.append(p4)
    f4, p4 = make_synthetic_features(n_contexts=nc, n_actions=na, dim=12,
        context_generation="bernoulli", feature_expansion="none", seed=10*seed_problem)
    print(f4.shape, p4.shape)
    rep_list.append(f4*9)
    param_list.append(p4)
    f4, p4 = make_synthetic_features(n_contexts=nc, n_actions=na, dim=12,
        context_generation="gaussian", feature_expansion="none", seed=10*seed_problem)
    print(f4.shape, p4.shape)
    rep_list.append(f4)
    param_list.append(p4)

print(f"total rep: {len(rep_list)}")

np.save(out_file, np.array([rep_list, param_list, reference_rep], dtype=object), allow_pickle=True)
a,b,c = np.load(out_file, allow_pickle=True)

print(np.all([np.allclose(rep_list[i], a[i]) for i in range(len(rep_list))]))
print(np.all([np.allclose(param_list[i], b[i]) for i in range(len(param_list))]))
print(c == reference_rep)
