import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='oful0')
args = parser.parse_args()

os.chdir('../logs') 
df = pd.read_csv(args.name + '.csv', index_col=False)
value = df['cumregret']
plt.plot(range(len(value)), value)
plt.show()