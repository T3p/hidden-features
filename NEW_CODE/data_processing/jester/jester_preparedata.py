import numpy as np
from scipy.sparse.linalg import svds

# load data
data_path = "jester_data_40jokes_19181users.npy"

ratings = np.load(data_path)
print("Loaded dataset: {}".format(ratings.shape))

n_users, n_items = ratings.shape
ratings = ratings / 10  # normalize ratings
print("ratings: max {0} - min {1}".format(ratings.max(), ratings.min()))


# SVD
K = 36
U, s, Vt = svds(ratings, k=K)
s = np.diag(s)
U = np.dot(U, s)

print(f"users features (n_users x dim): {U.shape}")
print(f"arm features (n_arms x dim): {Vt.T.shape}")

# MSE
Yhat = U.dot(Vt)
rmse = np.sqrt(np.mean(np.abs(Yhat - ratings) ** 2))
print("K: ", K)
print("RMSE:", rmse)
print("MAX_ERR:", np.abs(Yhat - ratings).max())


# GENERATE DATASET
# fit large "ground-truth" network
X, y = [], []
for t in range(n_users):
    for z in range(n_items):
        feat = np.concatenate([U[t], Vt[:, z]]).ravel()
        X.append(feat)
        y.append(ratings[t, z])
X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)


np.savez_compressed(f"jester_svd{K}.npz", X=X, y=y, user_features=U, arm_features=Vt.T)
