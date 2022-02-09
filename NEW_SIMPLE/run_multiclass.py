import numpy as np
from multiclass import MulticlassToBandit
from sklearn.datasets import load_wine

Xx, Yy =load_wine(return_X_y=True)

env = MulticlassToBandit(X=Xx, y=Yy, dataset_name="wine", seed=0, noise="gaussian", noise_param=0.1)

horizon = 10
print(env.action_space)

for t in range(horizon):
    context = env.sample_context()
    action = np.random.randint(0, env.action_space.n, 1).item()
    reward = env.step(action)
    print(context, action, reward)
