import argparse
import numpy as np
import pandas as pd
import os

def parse_and_load_npz(filename: str) -> dict:
    """Load data from a result file and parse metadata from filename.

    Args:
        filename: Path to the .npz result file

    Returns:
        Dictionary containing loaded arrays and metadata
    """

    data = np.load(filename, allow_pickle=True)
    data = {k: v for k, v in data.items()}
    data.update(data.pop("args", np.array({})).item())  # Flatten args
    return data




def parse_folder(root_dir: str) -> pd.DataFrame:
    """Parse all .npz files in a folder and compile into a DataFrame.

    Args:
        folder: Path to the folder containing .npz files

    Returns:
        DataFrame with parsed data and metadata
    """

    records = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".npz"):
                filepath = os.path.join(dirpath, file)
                try:
                    record = parse_and_load_npz(filepath)
                    records.append(record)
                except Exception as e:
                    print(f"Error parsing {file}: {e}")

    df = pd.DataFrame(records)

    if df.empty:
        print("No valid .npz files found.")
        return df

    # drop columns starting with wandb, or including eval
    df = df[[col for col in df.columns if not (col.startswith("wandb") or "eval" in col)]]

    return df


def aggregate_data(df: pd.DataFrame, agg_cols: list[str]) -> pd.DataFrame:
    
    if not agg_cols:
        # drop final_scores and final_returns columns
        df = df.drop(columns=["final_scores", "final_returns"],
                     errors="ignore")
        return df

    agg_df = (
        df.groupby(agg_cols)
        .agg(
            {
                "final_scores_mean": ["mean", "std"],
                "final_scores_std": ["mean"],
                "final_returns_mean": ["mean", "std"],
                "final_returns_std": ["mean"],
            }
        )
        .reset_index()
    )

    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    numeric_cols = [c for c in agg_df.columns if c not in agg_cols]
    agg_df[numeric_cols] = agg_df[numeric_cols].apply(pd.to_numeric, errors="coerce").round(2)

    return agg_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse .npz result files and aggregate data.")

    parser.add_argument("folder", type=str, help="Path to the folder containing .npz result files")
    parser.add_argument("--aggregate-cols", nargs='+', default=[], help="Columns to aggregate by")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save the aggregated CSV file")

    args = parser.parse_args()

    df = parse_folder(args.folder)
    if df.empty:
        print("No data to aggregate.")
    else:
        agg_df = aggregate_data(df, args.aggregate_cols)
        print(agg_df)
        if args.output_csv:
            agg_df.to_csv(args.output_csv, index=False)
            print(f"Aggregated data saved to {args.output_csv}")




