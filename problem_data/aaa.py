import numpy as np
import json
A = np.load("basic_features.npy")
b = np.load("basic_param.npy")
print(A)
print(b)

data = {"features": A.tolist(), "param": b.tolist()}

with open("simple_rep.json", "w") as outf:
    json.dump(data, outf)
