import numpy as np
import sys
sys.path.insert(0, '..')
from lbrl.linearenv import LinearEnv, LinearRepresentation
from lbrl.hlsutils import derank_hls, hls_lambda, is_hls, reduce_dim, make_reshaped_linrep, hls_rank
import json
from lbrl.utils import make_synthetic_features

nc=1000
na=6
dim=10
seed_problem=2311
DECIMALS=2

min_gap = 0 
count = 0
EL = np.linspace(-2,2,21).astype(np.float32)
print(EL)
while min_gap < 0.1 and count<10000:
    count += 1
    features, param = make_synthetic_features(n_contexts=nc, n_actions=na, dim=dim,
        context_generation="bernoulli", feature_expansion="none", seed=seed_problem)
    # features = np.random.binomial(1,p=0.75,size=(nc,na,dim)).reshape(nc,na,dim)
    features = np.random.choice(EL, size=(nc,na,dim))
    features = np.round(features, decimals=DECIMALS)
    # param = np.random.binomial(1,p=0.9,size=dim)
    param = np.random.choice([-1,0,1], p=[0.4,0.2,0.4], size=dim)
    true_reward=features @ param

    # compute gap
    min_gap = np.inf
    min_gap_ctx = []
    for ctx in range(true_reward.shape[0]):
        rr = true_reward[ctx]
        arr = sorted(rr)
        assert len(arr) == na
        gap = np.inf
        for i in range(na-1):
            diff = arr[na-1] - arr[i]
            if np.abs(diff-min_gap) < 1e-6:
                min_gap_ctx = [ctx]
            elif diff < min_gap and diff > 1e-6:
                min_gap = diff
                min_gap_ctx = [ctx]
    
    if  min_gap > 0.1:
        print(f"min_gap: {min_gap}")
        print(f"min_gap_ctx rewards: {true_reward[min_gap_ctx]}")

        print(f"dim: {dim}")
        print(f"feature norm: {np.linalg.norm(features,2,axis=-1).max()}")
        print(f"param norm: {np.linalg.norm(param,2)}")
        current_reward = features @ param
        error = np.abs(current_reward - true_reward).max()
        print(f"min gap: {min_gap}")
        print(f"realizable: {error < min_gap}")
        print(f"error: {error}")
        print(f"is HLS: {is_hls(features, true_reward, tol=1e-6)}")
        print(f"HSL rank: {hls_rank(features, true_reward, tol=1e-6)}")
        print(f"lambda HLS: {hls_lambda(features, true_reward)}")
        # print(true_reward)

        print(f"param: {param}")

        rep_list = [features]
        param_list = [param]
        position_reference_rep = 0

out_file = "categorical_largegap.npy"
np.save(out_file, np.array([rep_list, param_list, position_reference_rep], dtype=object), allow_pickle=True)
a,b,c = np.load(out_file, allow_pickle=True)

print(np.all([np.allclose(rep_list[i], a[i]) for i in range(len(rep_list))]))
print(np.all([np.allclose(param_list[i], b[i]) for i in range(len(param_list))]))
print(c == position_reference_rep)