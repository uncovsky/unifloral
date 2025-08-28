import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
import re

def parse_and_load_npz(filename: str) -> dict:
    """Load data from a result file and parse metadata from filename.

    Args:
        filename: Path to the .npz result file

    Returns:
        Dictionary containing loaded arrays and metadata
    """
    # Parse filename to extract algorithm, dataset, and timestamp
    pattern = r"([^_]+(?:_[^_]+)?)_(.+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    match = re.match(pattern, os.path.basename(filename))
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")

    algorithm, dataset, dt_str = match.groups()
    dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")

    data = np.load(filename, allow_pickle=True)
    data = {k: v for k, v in data.items()}
    data["algorithm"] = algorithm
    data["dataset"] = dataset
    data["datetime"] = dt
    data.update(data.pop("args", np.array({})).item())  # Flatten args
    return data




def parse_folder(folder: str) -> pd.DataFrame:
    """Parse all .npz files in a folder and compile into a DataFrame.

    Args:
        folder: Path to the folder containing .npz files

    Returns:
        DataFrame with parsed data and metadata
    """

    records = []
    for file in os.listdir(folder):
        if file.endswith(".npz"):
            filepath = os.path.join(folder, file)
            try:
                record = parse_and_load_npz(filepath)
                records.append(record)
            except Exception as e:
                print(f"Error parsing {file}: {e}")


    df = pd.DataFrame(records)

    # drop columns starting with wandb, or including eval
    df = df[[col for col in df.columns if not (col.startswith("wandb") or "eval" in col)]]

    return df



def generate_square_reach_figure(df: pd.DataFrame, save_path: str):

    # Want to generate the following figure:
    # Group by algorithm and hyperparameter settings (everything except dataset
    # and seed)

    # Make a figure with results from square reach 10, 15, 20 on X axis
    # (increasing horizon)


    fig, ax = plt.subplots(figsize=(10, 6))


    filter_columns = [
            "final_returns",
            "final_scores",
            "log",
            "datetime"
    ]

    # Cast all to strings
    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].astype(str)
    df = df[[col for col in df.columns if col not in filter_columns]]

    group_cols = [col for col in df.columns if col not in ["seed", 
                                                           "final_returns_mean", 
                                                           "final_returns_std"]]

    df[group_cols] = df[group_cols].fillna("MISSING")

    df["final_returns_mean"] = df["final_returns_mean"].astype(float)

    grouped = df.groupby(group_cols).agg({
        "final_returns_mean": ["mean", "std"],
        "seed": "count"
    }).reset_index()


    print(grouped.keys())

    cql_params = ['cql_temperature', 'cql_min_q_weight']
    shared_params = ['num_critics']
    independent_params = ['num_critics', 'cql_min_q_weight', 'actor_lcb_coef']

if __name__ == "__main__":
    res = parse_folder("../results/square_reach_5-15/")
    independent_params = ['num_critics', 'cql_min_q_weight', 'actor_lcb_coef']
    print(res.groupby(independent_params + ["dataset"])["final_returns_mean"].mean())

    # print algorithm name of nonzero final_returns_mean
    #generate_square_reach_figure(res, "square_reach.png")
