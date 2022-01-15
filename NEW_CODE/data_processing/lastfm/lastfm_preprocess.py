from pandas.core.arrays import categorical
import requests
import os
import pandas as pd
import numpy as np
from zipfile import ZipFile

for remote_url in [ 
    "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
]:

    local = remote_url.split("/")[-1]
    if not os.path.isfile(local):
        print(f"downloading {remote_url}")
        data = requests.get(remote_url)
        with open(local, 'wb')as file:
            file.write(data.content)

with ZipFile('hetrec2011-lastfm-2k.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

artist_threshold = 70
user_threshold = 10

df = pd.read_csv("user_artists.dat", sep="\s+")

tmp = df.groupby(by=['artistID']).userID.nunique()
artists = tmp[tmp > artist_threshold].index
# artists

filtered = df[df.artistID.isin(artists)]
tmp = filtered.groupby(by=['userID']).artistID.nunique()
users = tmp[tmp > user_threshold].index
# users

final = filtered[filtered.userID.isin(users)]
# final

tmp = final.pivot_table(index="userID", columns="artistID", values="weight", fill_value=0)
matrix = tmp.to_numpy()
print(f"data [users x actions]: {matrix.shape}")

norm_matrix = matrix / np.max(matrix)
log_matrix = np.log(1+matrix)
norm_log_matrix = log_matrix / np.max(log_matrix)
np.save('lastfm', norm_matrix)
np.save('lastfmlog', norm_log_matrix)