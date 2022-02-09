import numpy as np
from multiclass import MulticlassToBandit
from sklearn.datasets import load_wine
from torchleader_discrete import TorchLeaderDiscrete, Critic, TorchLinUCBDiscrete
import matplotlib.pyplot as plt

Xx, Yy = load_wine(return_X_y=True)

env = MulticlassToBandit(X=Xx, y=Yy, dataset_name="wine", seed=0, noise="gaussian", noise_param=0.1)

net = Critic(dim_context=Xx.shape[1], dim_actions=env.action_space.n)
agent = TorchLinUCBDiscrete(
    env=env,
    net=net,
    batch_size=64,
    max_epochs=3,
    update_every_n_steps=1,
    learning_rate=0.01,
    weight_l2param=1,
    weight_l2features=0,
    weight_mse=1,
    weight_spectral=0,
    bonus_scale=1,
    )

agent.reset()
output = agent.run(2000)

plt.plot(output['regret'])
plt.show()
