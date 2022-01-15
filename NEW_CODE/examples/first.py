from examples.finite_representations import is_hls
import numpy as np
import xb
from xb.envs.contextualfinite import ContextualFinite
from xb.envs.synthetic import LinearRandom, LinearCB
from xb.runner import Runner
from xb.algorithms import LinUCB, BatchedLinUCB
from xb.algorithms import LinLEADER, BatchedLinLEADER, LinLEADERElim
from xb.algorithms.linleaderselect import LinLeaderSelect, LinLeaderElimSelect
from xb.algorithms.regbased import RegBasedCBFeatures
import matplotlib.pyplot as plt
import xb.envs.synthetic.linutils as linutils
from copy import deepcopy
from joblib import dump, load
from itertools import cycle
import torch
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

class VNet(torch.nn.Module):
    def __init__(self, obs_size):
        super(VNet, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 1)
        # self.relu = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(64, 256)
        # self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, input):
        # x = self.relu(self.fc1(input))
        # x = self.relu(self.fc2(x))
        # value = self.fc3(x)
        value = self.fc1(input)
        return value

T = 10000
SEED = 97764652
np.random.seed(SEED)

# env = LinearRandom(n_contexts=10, n_actions=5, feature_dim=3, disjoint=False, random_state=1)
nc, na, dim = 100, 5, 10
features, param = linutils.make_random_linrep(
    n_contexts=nc, n_actions=na, feature_dim=dim, 
    ortho=True, normalize=True, seed=SEED, method="gaussian")
true_rewards = features @ param
rep_list = []
param_list = []
for i in range(1, dim):
    fi, pi = linutils.derank_hls(features=features, param=param, newrank=i, transform=True, normalize=True)
    if np.random.binomial(1, p=0.1):
        print(f"adding random noise to rep {i-1}")
        fi = fi + np.random.randn(*fi.shape)
    rep_list.append(ContextualFinite._rep(fi))
    param_list.append(pi)
rep_list.append(ContextualFinite._rep(features))
param_list.append(param)

for i in range(len(rep_list)):
    print()
    print(f"features_rank{i}: {linutils.features_rank(rep_list[i].features)}")
    print(f"hls_rank{i}: {linutils.hls_rank(rep_list[i].features, param_list[i])}")
    print(f"is_hls{i}: {linutils.is_hls(rep_list[i].features, param_list[i])}")
    print(f"pred_error{i}: {linutils.pred_error(rep_list[i].features, param_list[i], true_rewards)}")
print()

env = LinearCB(features=features, rewards=true_rewards, param=param, random_state=SEED)
# env = xb.load("jester_lin", data_path="data_processing", random_state=0)

ALGS = [
    LinUCB(
        rep=rep_list[-1], reg_val=1., noise_std=1., features_bound=2, param_bound= 2, bonus_scale=.1, delta= 0.01, adaptive_ci=True, random_state=0,
        # batch_type="fix", batch_param=int(np.sqrt(T).item())
    ),
    LinLEADER(
        reps=rep_list, reg_val=1., noise_std=1., features_bound=2, param_bound= 2, bonus_scale=.1, delta= 0.01, adaptive_ci=True, random_state=0,
        # batch_type="fix", batch_param=int(np.sqrt(T).item())
    ),
    LinLeaderSelect(
        reps=rep_list, reg_val=1., noise_std=1., features_bound=2, param_bound= 2, bonus_scale=.1, delta= 0.01, adaptive_ci=True, random_state=0,
        time_multiplier_update=1
        # batch_type="fix", batch_param=int(np.sqrt(T).item())
    ),
    # LinLeaderElimSelect(
    #     reps=rep_list, reg_val=1., noise_std=1., features_bound=2, param_bound= 2, bonus_scale=.1, delta= 0.01, adaptive_ci=True, random_state=0,
    #     # batch_type="fix", batch_param=int(np.sqrt(T).item())
    # ),
    # RegBasedCBFeatures(
    #     rep=rep_list[-1], 
    #     model_constructor=lambda : VNet(rep_list[-1].features_dim()), 
    #     optimizer_constructor=None, 
    #     memory_capacity=int(T), 
    #     epochs=30,
    #     batch_size=64, device="cpu", seed=SEED,
    #     exploration="igw",
    #     igw_forcing=na, igw_mult=1.
    # )
]
REGRETS = []

for i, algo in enumerate(ALGS):
    name = type(algo).__name__
    print(f"Running {name}...")
    runner = Runner(env=deepcopy(env), algo=algo, T_max=T)
    runner.reset()
    out = runner()
    REGRETS.append(out)
    plt.plot(out['regret'], next(linecycler), label=name)
    # dump(runner, 'alg{i}.joblib') 

# clf = load('runner.joblib') 
# clf.T_max = int(clf.T_max * 1.1)
# out2 = clf()

plt.legend()

plt.figure()
h = ALGS[2]
u = np.array(h.mineig)
for i in range(u.shape[1]):
    plt.plot(np.arange(u.shape[0]), u[:,i])
plt.show()

