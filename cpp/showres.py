import numpy as np 
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import tikzplotlib
import sys

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
MARKEVERY = 10
MARKSIZE = 4

markers=['o', 's', 'x', '*', '^', 'D', '|', 'v', 'h']
i = 0

#directory = 'build/'
directory = './'
for filename in sorted(os.listdir(directory)):
    if filename.endswith("_pseudoregrets.csv.gz"):
        print(filename)
        algo_name = filename.split('_')[0]
        algo_name = algo_name.replace('#', '_')

        if algo_name.startswith(r"\algo"):
            kwargs = {'linestyle':'dashed', 'linewidth':3}
        else:
            kwargs = {'linestyle':'solid', 'linewidth':2, 'marker':markers[i%len(markers)], 
                      'markevery':MARKEVERY, 'markerfacecolor':'none',
                      'markersize':MARKSIZE}
            i+=1

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
#plt.xscale("log")
plt.legend()

#Style
plt.xlabel(r'Rounds $n$')
plt.ylabel(r'\emph{Pseudo} Regret')

if len(sys.argv)>1:    
    figname = sys.argv[1]
else:
    figname = 'fig2'
print(figname)
plt.savefig(figname+'.png')
tikzplotlib.save(figname+".tex")
plt.show()
