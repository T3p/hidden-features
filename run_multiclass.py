import numpy as np
from multiclass import MulticlassToBandit, DiscreteFix
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from torchleader_discrete import TorchLeaderDiscrete, Critic, TorchLinUCBDiscrete
import matplotlib.pyplot as plt
import openml

# example: https://openml.github.io/openml-python/develop/examples/30_extended/datasets_tutorial.html
# This is done based on the dataset ID.
dataset = openml.datasets.get_dataset(1471)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:500])

Xx, yy, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)


# Xx, yy = load_wine(return_X_y=True)

env = MulticlassToBandit(X=Xx, y=yy, dataset_name=None, seed=0, noise="gaussian", noise_param=0.1)

assert isinstance(env.action_space, DiscreteFix)

# enc = OneHotEncoder(sparse=False)
# enc.fit(np.arange(env.action_space.n).reshape(-1,1))
# dim_actions = enc.transform(np.arange(env.action_space.n).reshape(-1,1)).shape[1]
# print(Xx.shape[1], dim_actions)
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
    use_onehotencoding=True
    )

agent.reset()
output = agent.run(2000)

plt.plot(output['regret'])
plt.show()
