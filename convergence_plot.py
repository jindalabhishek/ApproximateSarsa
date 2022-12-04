import os

import pandas as pd
import matplotlib.pyplot as plt

dfs = []
for file in os.listdir('ApproximateSarsaAgent'):
    df = pd.read_csv('ApproximateSarsaAgent/'+file)
    dfs.append(df)

df_concat = pd.concat(dfs)
by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()

averaged_run_data = sum(dfs) / len(dfs)
averaged_run_data = averaged_run_data.rolling(100).mean()

averaged_run_data.plot()
plt.show()
print(averaged_run_data.head())