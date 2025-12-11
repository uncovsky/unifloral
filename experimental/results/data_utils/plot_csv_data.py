import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_individual(csv_path: Path, out_root: Path):

    """
        For each project exported into a csv (and each metric considered)
        create a plot which included every run individually.
    """
    df = pd.read_csv(csv_path)

    # require _step for x-axis
    if "_step" not in df.columns:
        raise ValueError(f"{csv_path} does not contain '_step' column for x-axis.")

    # group by run index
    run_group_col = "run_index"

    # Metrics to plot (exclude bookkeeping)
    exclude = {"_step", run_group_col, "_timestamp", "_runtime", "run_id", "run_name", "run_index"}

    metrics = [c for c in df.columns if c not in exclude]

    csv_folder = out_root / csv_path.stem
    ensure_dir(csv_folder)

    # For each metric, plot each run (Matplotlib will cycle colors automatically)
    grouped = list(df.groupby(run_group_col))
    for metric in metrics:
        plt.figure()

        # Add each run
        for run_id, sub in grouped:
            if metric not in sub:
                continue

            x = sub["_step"]
            y = sub[metric]
            plt.plot(x, y, label=f"Run {run_id}", linewidth=1)

        plt.title(f"{csv_path.stem} — {metric} (individual runs)")
        plt.xlabel("_step")
        plt.ylabel(metric)
        n_runs = len(grouped)
        plt.legend(fontsize="x-small", loc="best", ncol=2 if n_runs > 8 else 1)
        plt.grid(True)
        plt.tight_layout()
        fname = csv_folder / f"{metric}_individual.png"
        plt.savefig(fname)
        plt.close()

def aggregate_and_shared_plot(csv_paths, out_root: Path):
    """
        Aggregate metrics across runs for each algorithm and display them along
        std bars in a shared plot. 
    """
    # For each csv (algorithm), we collect per-run data, align on _step, compute mean/std per metric
    alg_aggregates = {}  # alg_name -> dict(metric -> DataFrame with columns step, mean, std)
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "_step" not in df.columns:
            raise ValueError(f"{csv_path} missing _step column.")
        alg_name = csv_path.stem  # used as label
        
        # group by run index
        run_group_col = "run_index"

        metrics = [c for c in df.columns if c not in ("_step", run_group_col, "_timestamp", "_runtime", "run_id", "run_name", "run_index")]

        # Build per-run pivoted frames
        per_metric_agg = {}
        for metric in metrics:
            # Collect all runs into DataFrames indexed by step
            runs = []
            for run_id, sub in df.groupby(run_group_col):
                tmp = sub[["_step", metric]].rename(columns={metric: str(run_id)})
                tmp = tmp.set_index("_step")
                runs.append(tmp)
            if not runs:
                continue
            concat = pd.concat(runs, axis=1)  
            concat = concat.sort_index()
            mean = concat.mean(axis=1)
            std = concat.std(axis=1)
            per_metric_agg[metric] = pd.DataFrame({
                "_step": mean.index,
                "mean": mean.values,
                "std": std.values,
            })
        alg_aggregates[alg_name] = per_metric_agg

    all_metrics = set()
    for per in alg_aggregates.values():
        all_metrics.update(per.keys())

    shared_folder = out_root / "shared_plots"
    ensure_dir(shared_folder)

    for metric in sorted(all_metrics):
        plt.figure()
        for alg_name, per in alg_aggregates.items():
            if metric not in per:
                continue
            dfm = per[metric]
            x = dfm["_step"]
            mean = dfm["mean"]
            std = dfm["std"]
            plt.plot(x, mean, label=f"{alg_name}_mean")
            # shaded std
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.title(f"Shared aggregated metric: {metric}")
        plt.xlabel("_step")
        plt.ylabel(metric)
        plt.legend(fontsize="small", loc="best")
        plt.grid(True)
        fname = shared_folder / f"{metric}_shared.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()


def plot_bias_estimates_avg(csv_path: Path, out_root: Path):
    """ 
        Bias estimates need to be treated a little differently.
        They are aggregated across runs of a single algorithm and displayed
        alongside each other (see how pessimism varies with more OOD actions)
    """

    df = pd.read_csv(csv_path)
    if "_step" not in df.columns:
        raise ValueError(f"{csv_path} missing '_step' column.")

    # Determine run grouping
    run_group_col = "run_index"

    # Filter bias_estimate_* metrics
    bias_metrics = sorted([c for c in df.columns if c.startswith("bias_estimate_")])

    if not bias_metrics:
        return  # nothing to do

    # Prepare aggregated (mean/std) per metric
    agg_per_metric = {}
    for metric in bias_metrics:
        runs = []
        for run_id, sub in df.groupby(run_group_col):
            tmp = sub[["_step", metric]].rename(columns={metric: str(run_id)})
            tmp = tmp.set_index("_step")
            runs.append(tmp)
        if not runs:
            continue
        concat = pd.concat(runs, axis=1).sort_index()
        concat = concat.interpolate(method="index", limit_direction="both")
        mean = concat.mean(axis=1)
        std = concat.std(axis=1)
        agg_per_metric[metric] = pd.DataFrame({
            "_step": mean.index,
            "mean": mean.values,
            "std": std.values,
        })

    # Plot all bias_estimate_* means (with shaded std) in one figure
    csv_folder = out_root / csv_path.stem
    ensure_dir(csv_folder)
    plt.figure()
    for metric in bias_metrics:
        if metric not in agg_per_metric:
            continue
        dfm = agg_per_metric[metric]
        x = dfm["_step"]
        mean = dfm["mean"]
        std = dfm["std"]
        plt.plot(x, mean, label=f"{metric}_mean", linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.title(f"{csv_path.stem} — averaged bias_estimates across runs")
    plt.xlabel("_step")
    plt.ylabel("bias_estimate")
    plt.legend(fontsize="small", loc="best")
    plt.grid(True)
    plt.tight_layout()
    fname = csv_folder / "bias_estimates_avg_across_runs.png"
    plt.savefig(fname)
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="Process CSV metric files into individual and shared plots.")
    parser.add_argument("input_dir", type=Path, help="Directory containing CSV files.")
    parser.add_argument("output_dir", type=Path, help="Directory to write plots into.")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    ensure_dir(output_dir)

    csv_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in (".csv",)])
    if not csv_paths:
        print(f"No CSVs found in {input_dir}")
        return

    for csv_path in csv_paths:
        print("Processing:", csv_path.name)
        try:
            plot_individual(csv_path, output_dir)
            plot_bias_estimates_avg(csv_path, output_dir)
        except Exception as e:
            print(f"Failed processing {csv_path.name}: {e}")
    try:
        aggregate_and_shared_plot(csv_paths, output_dir)
    except Exception as e:
        print(f"Failed shared aggregation: {e}")

if __name__ == "__main__":
    main()
