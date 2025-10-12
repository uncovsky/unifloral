import argparse
import numpy as np
import os
import sys
import signal
import time
import yaml
import wandb

from helper_scripts.experiment_utils import sweep_folder, load_configs, get_param


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

    args = parser.parse_args()

    gpu_num = args.gpu
    print(f"Using GPU {gpu_num}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    


    """
    "experiments/diversity_experiments/std_ood_data",
    "experiments/diversity_experiments/shared_indep_targets",
    "experiments/diversity_experiments/prior_vs_edac",

    "experiments/bias_experiments/sac_n_expert",
    "experiments/bias_experiments/pbrl_cql_values",

    "experiments/bandit_experiments/configs",
    "experiments/reachability_experiments/",
    """

    # All the experiments in the thesis
    experiment_folders = [
        "experiments/unified_experiments/",
    ]

    unifloral_folders = [
        "experiments/unifloral_eval/",
    ]


    for experiment_folder in experiment_folders:
        print(f"Running sweeps in {experiment_folder}")
        break

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
                sweep_folder(
                    unifloral_folder,
                    entity=args.entity,
                    project=args.project,
                    run_limit=args.run_limit,
                    environment=env
                )




