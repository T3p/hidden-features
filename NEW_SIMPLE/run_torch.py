# import hydra
# from omegaconf import DictConfig

from utils_linear import make_random_linrep
from linearenv import LinearEnv, LinearRepresentation
from torchleader import TorchLeader, XNet
# from litleader import LitLeader, XNet
import numpy as np

import matplotlib.pyplot as plt

# @hydra.main(config_path=".", config_name="config")
# def my_app(cfg: DictConfig) -> None:
#     pass

if __name__ == "__main__":
#     my_app()
    SEED = 0
    NOISE = 0.5
    nc, na, dim = 100, 5, 10
    features, param = make_random_linrep(
        n_contexts=nc, n_actions=na, feature_dim=dim, 
        ortho=True, normalize=True, seed=SEED, method="gaussian")

    env = LinearEnv(features=features, param=param, rew_noise=NOISE)
    rep = LinearRepresentation(env.features)
    net = XNet(dim_input=dim)
    T=10000
    algo = TorchLeader(
        env=env, net=net, representation=rep, noise_std=NOISE,
        param_bound=np.linalg.norm(env.param,2), features_bound=np.linalg.norm(env.features,2, axis=-1).max(),
        delta=0.01,
        random_state=SEED, device="cpu", batch_size=64, max_epochs=3,
        weight_l2param=1.,
        weight_mse=0.3, weight_spectral=1, weight_l2features=0,
        buffer_capacity=T, update_every_n_steps=1,
        learning_rate=0.01
    )
    algo.reset()
    output = algo.run(horizon=T)

    plt.plot(output['regret'])
    plt.show()
