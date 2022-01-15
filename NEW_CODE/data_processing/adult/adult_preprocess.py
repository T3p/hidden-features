from pandas.core.arrays import categorical
import requests
import os
import pandas as pd
import numpy as np

for remote_url in [ 
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    # "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    # "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
]:

    local = remote_url.split("/")[-1]
    if not os.path.isfile(local):
        print(f"downloading {remote_url}")
        data = requests.get(remote_url)
        with open(local, 'wb')as file:
            file.write(data.content)

columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]

df = pd.read_csv("adult.data", header=None, na_values="?", skipinitialspace=True, names=columns, index_col=False)
print(df.shape)
df = df.dropna().reset_index()
print(df.head())

# print(df["occupation"].unique())
# print(len(df["occupation"].unique()))
# print(df)

cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

new_df = pd.DataFrame()
for col in columns:
    if col != "occupation":
        if col in cat_cols:
            y = pd.get_dummies(df[col], prefix=col)
        else:
            y = pd.DataFrame( {col: df[col].values}, index=df.index)
        # print(y.columns)
        # print(col, new_df.shape,y.shape)
        new_df = new_df.join(y) if not new_df.empty else y
# print(new_df.info())
print(new_df.shape)
print(new_df.info())

labels = df["occupation"].values
print(labels)

np.savez_compressed("adult_hotencoding", contexts=new_df.values, classes=labels)
