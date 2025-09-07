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

    if df.empty:
        print("No valid .npz files found.")
        return df

    # drop columns starting with wandb, or including eval
    df = df[[col for col in df.columns if not (col.startswith("wandb") or "eval" in col)]]

    return df
