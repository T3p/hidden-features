import numpy as np 
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import tikzplotlib


print("\n=================")
print(" running showres ")
print("=================\n")

# with open("build/pretty.json", "r") as inpu:
#     data = json.load(inpu)

# print(data)
# A = np.array(data['features'])
# p = np.array(data['param'])
# print(A.shape)
# print(p)
# exit(9)

EVERY = 100

#directory = 'build/'
directory = './'
for filename in sorted(os.listdir(directory)):
    if filename.endswith("_pseudoregrets.csv.gz"):
        print(filename)
        algo_name = filename.split('_')[0]

        if algo_name.startswith("LEADER"):
            kwargs = {'linestyle':'dashed', 'linewidth':3}
        elif algo_name.startswith("OFULBALELIM"):
            kwargs = {'linestyle':'dotted', 'linewidth':2}
        elif algo_name.startswith("OFULBAL"):
            kwargs = {'linestyle': (0, (1, 10)), 'linewidth':2} # loosely dotted
        else:
            kwargs = {'linestyle':'solid', 'linewidth':2}

        # A = np.genfromtxt(os.path.join(directory, filename), delimiter=',')
        A = pd.read_csv(filename, compression='gzip',  header=None).values
        if len(A.shape) == 1:
            A = A.reshape(-1,1)
        A = np.cumsum(A, axis=0)
        std = np.std(A, axis=1) / np.sqrt(A.shape[1])
        A = np.mean(A, axis=1)
        x = np.arange(A.shape[0])
        plt.plot(x[::EVERY], A[::EVERY], label=algo_name, **kwargs)
        plt.fill_between(x[::EVERY], A[::EVERY] - 2*std[::EVERY], A[::EVERY]+2*std[::EVERY], alpha=0.2)
plt.legend()
plt.savefig('fig2.png')
tikzplotlib.save("fig2.tex")
plt.show()
