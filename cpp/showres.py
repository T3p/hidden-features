import numpy as np 
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

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

#directory = 'build/'
directory = './'
for filename in sorted(os.listdir(directory)):
    if filename.endswith("_pseudoregrets.csv.gz"):
        print(filename)
        algo_name = filename.split('_')[0]

        if algo_name.startswith("MMOFUL"):
            kwargs = {'linestyle':'dashed', 'linewidth':3}
        elif algo_name.startswith("OFULBALELIM"):
            kwargs = {'linestyle':'dotted'}
        elif algo_name.startswith("OFULBAL"):
            kwargs = {'linestyle': (0, (1, 10))} # loosely dotted
        else:
            kwargs = {'linestyle':'solid'}

        # A = np.genfromtxt(os.path.join(directory, filename), delimiter=',')
        A = pd.read_csv(filename, compression='gzip',  header=None).values
        A = np.cumsum(A, axis=0)
        std = np.std(A, axis=1) / np.sqrt(A.shape[1])
        A = np.mean(A, axis=1)
        plt.plot(A, label=algo_name, **kwargs)
        plt.fill_between(np.arange(A.shape[0]), A - 2*std, A+2*std, alpha=0.2)
plt.legend()
plt.savefig('fig2.png')
plt.show()
