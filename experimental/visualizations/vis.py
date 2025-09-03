import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from datetime import datetime
import re
import seaborn as sns


from vis_utils import parse_and_load_npz, parse_folder


def process_square_reach_data(df: pd.DataFrame):
    print("Processing square-reach data...")
    # relevant hyperparams for different algorithms
    cql_params = ['cql_temperature', 'cql_min_q_weight']
    shared_params = ['num_critics']
    independent_params = ['num_critics', 'cql_min_q_weight', 'actor_lcb_coef']

    params = {}
    params["cql"] = cql_params
    params["shared_targets"] = shared_params
    params["independent_targets"] = independent_params

    fig, ax = plt.subplots(figsize=(10, 6))

    # unused columns to filter out
    filter_columns = [
            "final_returns",
            "final_scores",
            "log",
            "datetime"
    ]

    # Cast all to strings and filter
    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].astype(str)
    df = df[[col for col in df.columns if col not in filter_columns]]


    df["final_returns_mean"] = df["final_returns_mean"].astype(float)

    algorithms = df['algorithm'].unique()


    # algo name+hyperparams, dataset1_mean, dataset2_mean, ...

    rows = []

    for algo in algorithms:
        if algo not in params:
            print(f"Algorithm {algo} not in params, skipping!")
            continue
        
        algo_df = df[df['algorithm'] == algo]
        hyperparams = params[algo]

        # Remove duplicit runs logged with same hyperparams, seed, etc.
        algo_df = algo_df.drop_duplicates(subset=hyperparams + ['dataset_name', 'seed'])
        print(f"Processing {algo}, {len(algo_df)} entries")

        grouped = (
            algo_df.groupby(hyperparams + ['dataset_name'])
            .agg(final_returns_mean=('final_returns_mean', 'mean'))
            .reset_index()
        )

        
        # Build a string label: algo + hyperparam settings
        for group, row in grouped.iterrows():
            desc_parts = [f"{algo}"]
            desc_parts += [f"{hp}={row[hp]}" for hp in hyperparams]
            desc = ", ".join(desc_parts)

            text = row["dataset_name"]
            horizon = re.search(r"\d+", text)

            print(horizon)
            
            rows.append({
                "description": desc,
                "dataset": int(horizon.group()),
                "algorithm": algo,
                "final_returns_mean": row["final_returns_mean"]
            })

    long_df = pd.DataFrame(rows)

    assert not long_df.empty
    print(long_df)


    wide_df = long_df.pivot_table(
        index=["description", "algorithm"],
        columns="dataset",
        values="final_returns_mean"
    ).reset_index()


    # save to csv
    os.makedirs("csv", exist_ok=True)
    wide_df.to_csv("csv/square_reach_results.csv", index=False)
    return wide_df

def make_square_reach_figures(df: pd.DataFrame):
    """
    Creates per-algorithm and shared plots for square-reach results and saves them to 'figures/'.
    """

    # create figures directory
    os.makedirs("figures", exist_ok=True)
    make_square_reach_lineplot(df)
    make_square_reach_heatmap(df)



def make_square_reach_lineplot(df: pd.DataFrame):
    sns.set(style="whitegrid")  # nicer grid background
    palette = sns.color_palette("tab10")  # 10 distinct colors

    algorithms = df['algorithm'].unique()
    horizon_cols = sorted([col for col in df.columns if col not in ['description', 'algorithm']])
    x = np.arange(len(horizon_cols))
    # Plot per algorithm
    for algo in algorithms:
        plt.figure(figsize=(6,4))
        subset = df[df['algorithm'] == algo]
        for i, (_, row) in enumerate(subset.iterrows()):
            plt.plot(x, row[horizon_cols], marker='o', alpha=0.8, 
                     label=row['description'], color=palette[i % len(palette)], linewidth=2)
        plt.title(f"{algo} results", fontsize=14)
        plt.xlabel("Horizon", fontsize=12)
        plt.ylabel("Mean final result", fontsize=12)
        plt.xticks(x, ['']*len(x))
        plt.legend(fontsize=8)
        plt.tight_layout()
        
        # save
        safe_algo_name = re.sub(r'\W+', '_', algo)
        plt.savefig(f"figures/{safe_algo_name}_results.png", dpi=300)
        plt.close()

    # Shared plot with all algorithms
    plt.figure(figsize=(8,5))
    for i, (_, row) in enumerate(df.iterrows()):
        plt.plot(x, row[horizon_cols], marker='o', alpha=0.8, 
                 label=row['description'], color=palette[i % len(palette)], linewidth=2)
    plt.title("All algorithms", fontsize=14)
    plt.xlabel("Horizon", fontsize=12)
    plt.ylabel("Mean final result", fontsize=12)
    plt.xticks(x, ['']*len(x))
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("figures/all_algorithms_results.png", dpi=300)
    plt.close()


def make_square_reach_heatmap(df: pd.DataFrame):

    horizon_cols = sorted([col for col in df.columns if col not in ['description', 'algorithm']])
    
    os.makedirs("figures", exist_ok=True)
    
    heatmap_data = df.set_index('description')[horizon_cols]
    
    plt.figure(figsize=(10, max(6, len(df)*0.5)))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap=sns.color_palette("Blues", as_cmap=True), 
        cbar_kws={'label': 'Mean return'},
        linewidths=0.5,
        linecolor='white',
    )
    plt.title("Mean Return per Description and Horizon", fontsize=14)
    plt.xlabel("Horizon", fontsize=12)
    plt.ylabel("Description", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/heatmap_descriptions_horizons.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    res = parse_folder("../results/square_sweep/")
    df = process_square_reach_data(res)
    make_square_reach_figures(df)
