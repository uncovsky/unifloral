import argparse
import numpy as np
import os
import sys
import signal
import time
import yaml
import wandb

from helper_scripts.experiment_utils import sweep_folder, load_configs, get_param, wandb_sweep_from_config


"""
    Run all experiments sequentially on a given
    device
"""

if __name__ == "__main__":

    print("Ensuring data for experiments is collected...")
    # collect_data()

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0, help="idx of cuda device to use")
    parser.add_argument("--entity", type=str, default=None, help="Wandb entity to run the sweeps under, otherwise defaults to config value")
    parser.add_argument("--project", type=str, default=None, help="Wandb project to run the sweeps under, otherwise defaults to config value")
    parser.add_argument("--run_limit", type=int, default=None, help="Max number of runs to execute per sweep. Default None (unlimited)")
    parser.add_argument("--folder", type=str, default=None, help="If specified, only run sweeps in this folder")

    args = parser.parse_args()

    gpu_num = args.gpu
    print(f"Using GPU {gpu_num}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    """
    "experiments/diversity_experiments/std_ood_data",
    "experiments/diversity_experiments/shared_indep_targets",
    "experiments/diversity_experiments/prior_vs_edac",

    "experiments/bias_experiments/sac_n_expert",
    "experiments/bias_experiments/pbrl_cql_values",

    "experiments/bandit_experiments/configs",
    "experiments/reachability_experiments/",
    "experiments/unified_experiments/",
    """


    if args.folder is not None:
        print(f"Running sweeps in {args.folder}")
        sweep_folder(
            args.folder,
            entity=args.entity,
            project=args.project,
            run_limit=args.run_limit
        )

    # All the experiments in the thesis
    experiment_folders = [
        "experiments/bias_experiments/expert_vis",
    ]

    unifloral_folders = [
        #"experiments/unifloral_eval/",
    ]

    unifloral_runs = 10

    for experiment_folder in experiment_folders:
        print(f"Running sweeps in {experiment_folder}")

        sweep_folder(
            experiment_folder,
            entity=args.entity,
            project=args.project,
            run_limit=args.run_limit
        )


    for unifloral_folder in unifloral_folders:
        configs = load_configs(unifloral_folder)
        for config in configs:

            # Get datasets to run on
            datasets = get_param(config, "dataset-name")

            for env in datasets:
                print(f"Running unifloral sweep for {env}")
                wandb_sweep_from_config(
                    config,
                    entity=args.entity,
                    project=args.project,
                    run_limit=unifloral_runs,
                    environment=env
                )



