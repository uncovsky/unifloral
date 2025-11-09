import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from datetime import datetime
import re
import seaborn as sns


from vis_utils import parse_and_load_npz, parse_folder


def process_square_reach_data(df: pd.DataFrame):
    # save raw df
    os.makedirs("csv", exist_ok=True)
    returns = df[[col for col in df.columns if "final_returns" in col]]
    returns.to_csv("csv/square_reach_raw.csv", index=False)

    # relevant hyperparams for different algorithms
    params = [ "critic_lagrangian", "num_critics" ]

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
    df["final_returns_std"] = df["final_returns_std"].astype(float)

    algorithms = df['critic_regularizer'].unique()

    # algo name+hyperparams, dataset1_mean, dataset2_mean, ...
    rows = []

    for algo in algorithms:

        algo_df = df[df['critic_regularizer'] == algo]

        # Remove duplicit runs logged with same hyperparams, seed, etc.
        #algo_df = algo_df.drop_duplicates(subset=hyperparams + ['dataset_name', 'seed'])
        print(f"Processing {algo}, {len(algo_df)} entries")

        grouped = (
            algo_df.groupby(params + ['dataset_name'])
            .agg(
                final_returns_mean=('final_returns_mean', 'mean'),
                final_returns_std=('final_returns_std', 'mean')
            )
            .reset_index()
        )


        
        grouped.to_csv(f"csv/{algo}_grouped.csv", index=False)

        # Build a string label: algo + hyperparam settings
        for group, row in grouped.iterrows():
            desc_parts = [f"{algo}"]
            desc_parts += [f"{hp}={row[hp]}" for hp in params]
            desc = ", ".join(desc_parts)

            text = row["dataset_name"]
            match = re.search(r"horizon-(\d+)-eps(\d+)", text)

            eps = int(match.group(2))
            mapping = {1: "low", 2: "medium", 3: "high"}
            level = mapping.get(eps, "unknown")           

            dataset_name = "horizon-" + match.group(1) + "-" + level
            rows.append({
                "description": desc,
                "dataset": dataset_name,
                "horizon": int(match.group(1)),
                "level": level,
                "algorithm": algo,
                "final_returns_mean": row["final_returns_mean"],
                "final_returns_std": row["final_returns_std"],
            })

    print(rows)
    long_df = pd.DataFrame(rows)

    assert not long_df.empty


    wide_df = long_df.pivot_table(
        index=["description", "algorithm"],
        columns="dataset",
        values="final_returns_mean"
    ).reset_index()


    # save to csv
    os.makedirs("csv", exist_ok=True)
    wide_df.to_csv("csv/square_reach_results.csv", index=False)
    return long_df

def make_square_reach_figures(df: pd.DataFrame):
    """
    Creates per-algorithm and shared plots for square-reach results and saves them to 'figures/'.
    """

    # create figures directory
    os.makedirs("figures", exist_ok=True)
    make_square_reach_heatmap(df)


def format_algo(row):
    "translate back to hyperparams from desc"

    desc = row["description"]

    if row["algorithm"] == "cql":
        # alpha = cql_min_q_weight
        m = re.search(r"cql_min_q_weight=([\d.]+)", desc)
        alpha = m.group(1) if m else "?"
        return f"CQL (alpha={alpha})"

    elif row["algorithm"] == "independent_targets":
        # alpha = min_q_..., beta = lcb_..., N = num_critics
        m_alpha = re.search(r"cql_min_q_weight=([\w\d.]+)", desc)
        m_beta = re.search(r"actor_lcb_coef=([\w\d.]+)", desc)
        m_n = re.search(r"num_critics=(\d+)", desc)

        alpha = m_alpha.group(1) if m_alpha else "?"
        beta = m_beta.group(1) if m_beta else "?"
        N = m_n.group(1) if m_n else "?"

        return f"MSG (alpha={alpha}, beta={beta}, N={N})"

    elif row["algorithm"] == "shared_targets":
        # N = ensemble_size
        m_n = re.search(r"num_critics=(\d+)", desc)
        N = m_n.group(1) if m_n else "?"
        return f"SAC-(N={N})"

    else:
        return row["algorithm"]


def make_square_reach_heatmap(df: pd.DataFrame):
    df["algorithm_formatted"] = df.apply(format_algo, axis=1)

    df["col_label"] = df["horizon"].astype(str) + "-" + df["level"]

    pivot_mean = df.pivot_table(
            index="algorithm_formatted",
            columns="col_label",
            values="final_returns_mean",
            aggfunc="mean"
    )

    pivot_std = df.pivot_table(
        index="algorithm_formatted",
        columns="col_label",
        values="final_returns_std",
        aggfunc="mean"
    )

    # Sort columns by numeric horizon and level order
    level_order = ["low", "medium", "high"]
    def col_sort_key(col):
        horizon_str, lvl = col.split("-")
        return (int(horizon_str), level_order.index(lvl))

    pivot_mean = pivot_mean[sorted(pivot_mean.columns, key=col_sort_key)]
    pivot_std = pivot_std[sorted(pivot_std.columns, key=col_sort_key)]

    annot_text = pivot_mean.round(2).astype(str) + " ± " + pivot_std.round(2).astype(str)
    plt.rcParams.update({'font.size': 20, 'font.family': 'Dejavu Sans'})

    plt.figure(figsize=(20, 15))
    sns.heatmap(
        pivot_mean,                  # base colors by mean
        annot=annot_text,            # show mean ± std
        fmt="",
        cmap="Blues",
        cbar_kws={'label': 'Final Returns Mean'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 16}
    )

    plt.xlabel("Horizon / Level", fontsize=16, fontweight='bold')
    plt.ylabel(None)
    plt.title("Performance Heatmap", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold', rotation=45)
    plt.yticks(fontsize=14, fontweight='bold', rotation=45)
    plt.tight_layout()
    plt.savefig("figures/square_reach_heatmap.png", dpi=300, bbox_inches='tight') 
    print("Saved figure to figures/square_reach_heatmap.png")


if __name__ == "__main__":
    res = parse_folder("reach-results")
    df = process_square_reach_data(res)
    make_square_reach_figures(df)
