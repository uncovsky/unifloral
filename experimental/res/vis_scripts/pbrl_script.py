import pandas as pd

df = pd.read_csv('pbrl_runs.csv')

# drop where critic_lagrangian is 2.0
df = df[df['critic_lagrangian'] != 2.0]

# group by dataset, and count number of unique seeds
grouped = df.groupby('dataset_name')['seed'].nunique().reset_index()
print(grouped)
