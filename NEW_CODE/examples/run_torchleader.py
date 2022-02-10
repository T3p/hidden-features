import numpy as np
from xb.envs.synthetic import LinearRandom, LinearCB
from xb.runner import Runner
from xb.algorithms.torchleaderv0 import TorchLeader
import matplotlib.pyplot as plt
import xb.envs.synthetic.linutils as linutils
from copy import deepcopy
from joblib import dump, load
from itertools import cycle
import torch


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

nc, na, dim = 100, 5, 10
features, param = linutils.make_random_linrep(
    n_contexts=nc, n_actions=na, feature_dim=dim, 
    ortho=True, normalize=True, seed=SEED, method="gaussian")
true_rewards = features @ param

env = LinearCB(features=features, rewards=true_rewards, param=param, random_state=SEED)
algo = TorchLeader(
    env=env,
    representation=env.get_default_representation(), reg_val=1., noise_std=1., features_bound=2, param_bound= 2, bonus_scale=.1, delta= 0.01, adaptive_ci=True, random_state=0,
    reg_mse=1, reg_spectral=1, reg_norm=1
)

runner = Runner(env=env, algo=algo, T_max=T)
runner.reset()
out = runner(T)
plt.plot(out['regret'])
plt.show()
