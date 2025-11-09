import pandas as pd

# Load CSV
df = pd.read_csv("res_runs.csv")

# Hyperparameter columns that must match (except num_critics)
hp_cols = ["algorithm", "dataset_name", 
           "critic_lagrangian", "reg_lagrangian", 
           "actor_lcb_penalty"]

hp_cols = [
    "algorithm",
    "dataset_name",
    "critic_lagrangian",
    "reg_lagrangian",
    "actor_lcb_penalty"
]


# Group by all hyperparameters except num_critics
groups = df.groupby(hp_cols)

for hp_values, group in groups:
    # Keep only groups where num_critics varies
    if group["num_critics"].nunique() > 1:
        
        print("\n========================================================")
        print("Hyperparameter setting:")
        for name, val in zip(hp_cols, hp_values):
            print(f"  {name}: {val}")
        print("Different num_critics & their scores:\n")

        # Print each entry with its score row
        for _, row in group.iterrows():
            mean = row['final_scores_mean_mean']
            std = row['final_scores_mean_std']
            print(f"num_critics = {row['num_critics']}")
            print(f"  Score: {mean:.4f} ± {std:.4f}\n")
