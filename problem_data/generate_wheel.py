import numpy as np
import matplotlib.pyplot as plt
import pickle

mu_1 = 1.0
mu_2 = 0.8
mu_3 = 1.2
radius = 0.5

seed_problem = 0

rej = 0.05

ncontexts, narms, dim = 1000, 5, 7
features = np.zeros((ncontexts, narms, dim))
rewards = np.zeros((ncontexts, narms))
rewards[:, 0] = mu_1
rewards[:, 1:5] = mu_2

np.random.seed(seed_problem)
fig, ax = plt.subplots()
plt.plot([-1,-0.5],[0,0], color="black", alpha=0.5)
plt.plot([0.5,1],[0,0], color="black", alpha=0.5)
plt.plot([0,0],[-1,-0.5], color="black", alpha=0.5)
plt.plot([0,0],[0.5,1], color="black", alpha=0.5)
c1 = plt.Circle((0, 0), 1, color='black', fill=False, alpha=0.5)
c2 = plt.Circle((0, 0), 0.5, color='black', fill=False, alpha=0.5)
ax.add_patch(c1)
ax.add_patch(c2)
for i in range(ncontexts):
    x = None
    while x is None or radius - rej <= np.linalg.norm(x) <= radius + rej or (np.abs(x[0]) <= rej and radius <= np.abs(x[1]) <= 1) or (np.abs(x[1]) <= rej and radius <= np.abs(x[0]) <= 1):
        rho = np.random.rand()
        theta = np.random.rand() * 2 * np.pi
        x = np.array([np.cos(theta), np.sin(theta)]) * rho
    for j in range(narms):
        y = np.zeros(narms)
        y[j] = 1
        features[i,j,:] = np.concatenate([x,y])
    if np.linalg.norm(x) > radius:
        if x[0] >= 0 and x[1] >= 0:
            rewards[i, 1] = mu_3
            plt.scatter(x[0],x[1],c="red",s=10,alpha=0.5)
        elif x[0] >= 0 and x[1] < 0:
            rewards[i, 2] = mu_3
            plt.scatter(x[0],x[1],c="green",s=10,alpha=0.5)
        elif x[0] < 0 and x[1] >= 0:
            rewards[i, 3] = mu_3
            plt.scatter(x[0],x[1],c="orange",s=10,alpha=0.5)
        else:
            rewards[i, 4] = mu_3
            plt.scatter(x[0],x[1],c="purple",s=10,alpha=0.5)
    else:
        plt.scatter(x[0],x[1],c="blue",s=10,alpha=0.5)

with open(f"wheel_{ncontexts}_{rej}.pkl", "wb") as f:
    pickle.dump([features, rewards], f)

with open(f"wheel_{ncontexts}_{rej}.pkl", "rb") as f:
    a,b = pickle.load(f)

print(np.allclose(features, a))
print(np.allclose(rewards, b))
plt.show()
