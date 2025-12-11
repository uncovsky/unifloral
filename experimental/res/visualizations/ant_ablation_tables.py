import pandas as pd


df = pd.read_csv('../vis_data/antablation_final/results.csv')

# SAC 
sac_df = df[df['shared_targets'] == True]


print("SAC Results")
cols = ['dataset_name', 'critic_regularizer', 'critic_lagrangian',
        'num_critics',
        'final_scores_mean_mean', 'final_scores_mean_std']

sac_results = sac_df[cols][sac_df.critic_regularizer == 'msg']
print(sac_results)


print("PBRL Results")

pbrl_df = df[df['critic_regularizer'] == 'pbrl']
print(pbrl_df[cols])

print("msg Results")
msg_df = df[(df['critic_regularizer'] == 'msg') & (df['shared_targets'] == False)]
print(msg_df[cols + ['pi_operator']])

print("EDAC Results")
edac_df = df[(df['critic_regularizer'] == 'none')]
print(edac_df[cols + ['pi_operator']])



"""
    Ablation plots
"""


# Define the hyperparameters to optimize over
hyperparams = [
    'critic_regularizer',
    'critic_lagrangian',
    'ensemble_regularizer',
    'reg_lagrangian',
    'num_critics',
    'pi_operator',
    'shared_targets'
]

print(df.columns)
print("Best Hyperparameters per Dataset")
avg_scores = (
    df.groupby(hyperparams + ['dataset_name'], as_index=False)['final_scores_mean_mean']
      .mean()
      .rename(columns={'final_scores_mean_mean': 'mean_over_datasets'})
)

algorithm_def = [
    'critic_regularizer',
    'ensemble_regularizer',
    'reg_lagrangian',
    'num_critics',
    'pi_operator',
    'shared_targets'
]

# Columns to display
cols = algorithm_def + ['critic_lagrangian', 'dataset_name', 'final_scores_mean_mean', 'final_scores_mean_std']

# Select the best critic_lagrangian per algorithm per dataset
best_per_algo_dataset = (
    df.loc[
        df.groupby(algorithm_def + ['dataset_name'])['final_scores_mean_mean'].idxmax()
    ]
    .reset_index(drop=True)
)

print("Best critic_lagrangian per algorithm per dataset:")
print(best_per_algo_dataset[cols])
