import numpy as np
import sys
sys.path.insert(0, '..')
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import derank_hls, hls_lambda, is_hls, reduce_dim, hls_rank
from lbrl.utils import make_synthetic_features, inv_sherman_morrison
import json

ncontexts, narms, dim = 5000, 30, 20
contextgeneration = "bernoulli"
feature_expansion = "none"
seed_problem = 1931
n_nonrealizable = 0
NEW_REPS = [
    (None,5), #(dim, rank) dim=None=unchanged
    (None,10),
    (None,15),
    (15,5),
    (15,10),
    (15,15),
]
out_file = "nonhlsrealizable_c1.npy"

features, theta = make_synthetic_features(
    n_contexts=ncontexts, n_actions=narms, dim=dim,
    context_generation=contextgeneration, feature_expansion=feature_expansion,
    seed=seed_problem
)
problem_gen = np.random.RandomState(seed_problem)

rep_list = []
param_list = []
## add HLS
# rep_list.append(features)
# param_list.append(theta)
position_reference_rep = 0
for i in range(len(NEW_REPS)):
    newdim, newrank = NEW_REPS[i]
    print(newdim, newrank)
    fi = features.copy()
    pi = theta.copy()
    if newdim is not None:
        fi, pi = reduce_dim(features=fi, param=pi, newdim=newdim, transform=True, normalize=True, seed=seed_problem)
    fi, pi = derank_hls(features=fi, param=pi, newrank=newrank, transform=True, normalize=True, seed=seed_problem)
    # if np.random.binomial(1, p=0.1):
    #     print(f"adding random noise to rep {i-1}")
    #     fi = fi + np.random.randn(*fi.shape)
    rep_list.append(fi)
    param_list.append(pi)
true_reward = rep_list[position_reference_rep] @ param_list[position_reference_rep]

# non realizable
for i in range(n_nonrealizable):
    idx = problem_gen.choice(len(rep_list), 1).item()
    fi = rep_list[idx]
    # mask = np.random.binomial(1, p=0.5, size=fi.shape)
    fi = fi + problem_gen.randn(*fi.shape) * 0.6
    rep_list.append(fi)
    mtx, bv = np.eye(fi.shape[2])/0.0001, 0
    for kk in range(fi.shape[0]):
        for aa in range(fi.shape[1]):
            el = fi[kk,aa]
            mtx, _ = inv_sherman_morrison(el, mtx)
            bv = bv + true_reward[kk,aa] * el
    pi = mtx @ bv
    param_list.append(pi) # best fit to the true reward

min_gap = np.inf
for i in range(true_reward.shape[0]):
    rr = true_reward[i]
    sort_rr = sorted(rr)
    gap = sort_rr[-1] - sort_rr[-2]
    min_gap = min(gap, min_gap)
print(f"min gap: {min_gap}")

for i in range(len(rep_list)):
    print("\n")
    if i == position_reference_rep:
        print(f"Info representation({i}) [REFERENCE REP]")
    else:
        print(f"Info representation({i})")
    print(f"dim({i}): {rep_list[i].shape[2]}")
    print(f"feature norm({i}): {np.linalg.norm(rep_list[i],2,axis=-1).max()}")
    print(f"param norm({i}): {np.linalg.norm(param_list[i],2)}")
    current_reward = rep_list[i] @ param_list[i]
    error = np.abs(current_reward - true_reward).max()
    print(f"min gap: {min_gap}")
    print(f"realizable({i}): {error < min_gap}")
    print(f"error({i}): {error}")
    print(f"is HLS({i}): {is_hls(rep_list[i], true_reward)}")
    print(f"HLS rank({i}): {hls_rank(rep_list[i], true_reward)}")
    print(f"lambda HLS({i}): {hls_lambda(rep_list[i], true_reward)}")
print("\n")


np.save(out_file, np.array([rep_list, param_list, position_reference_rep], dtype=object), allow_pickle=True)
a,b,c = np.load(out_file, allow_pickle=True)

print(np.all([np.allclose(rep_list[i], a[i]) for i in range(len(rep_list))]))
print(np.all([np.allclose(param_list[i], b[i]) for i in range(len(param_list))]))
print(c == position_reference_rep)