import argparse
import numpy as np
import os
import sys
import signal
import time
import yaml
import wandb

from helper_scripts.experiment_utils import sweep_folder

# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

bias_exp_config_path = "./bias_experiments/configs/"

MUJOCO_TASKS = [
  "halfcheetah-medium-v2",
  "halfcheetah-expert-v2",
  "halfcheetah-medium-expert-v2",
]

ANT_TASKS = [
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
]

def get_param(cfg, key):
    entry = cfg["parameters"][key]
    if "value" in entry:
        return entry["value"]   
    elif "values" in entry:
        return entry["values"]            
    else:
        raise KeyError(f"No 'value' or 'values' for {key}")

def run_sweep(sweep_cfg, run_limit):

    algorithm_name = sweep_cfg.get("name", "default-algorithm")

    print(f"Creating sweep for algorithm: {algorithm_name}")

    steps = get_param(sweep_cfg, "num_updates")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    sweep_cfg["name"] = f"{algorithm_name}-{timestamp}"

    sweep_id = wandb.sweep(sweep_cfg)

    print(f"Sweep created with id: {sweep_id}, steps: {steps}")
    wandb.agent(sweep_id, function=None, count=run_limit)


def load_configs(config_dir):

    print("Loading configuration files...")
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
    
    if not config_files:
        print("No configuration files found in the specified directory.")
        return []

    configs = []
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        print(config_path)
        with open(config_path, "r") as f:
            try:
                sweep_cfg = yaml.safe_load(f)
                configs.append(sweep_cfg)

            except yaml.YAMLError as e:
                print(f"Error loading YAML file {config_file}: {e}")
                continue

    return configs


"""
    Utils for running unifloral style random sweeps with a fixed seed
"""
    

def sample_params(config, exclude_keys=None):
    """
    Given a sweep config, returns a dict of sampled parameters.

    Keys in exclude_keys or with fixed `value` are not sampled.
    Keys with 'min'/'max' are sampled uniformly from the interval.
    Keys with 'values' (categorical) are sampled uniformly.
    
    Args:
        config: dict, loaded from yaml
        exclude_keys: list of keys to never sample (default: seed, dataset-name)
    
    Returns:
        dict of sampled parameters
    """
    if exclude_keys is None:
        exclude_keys = ["seed", "dataset-name"]

    sampled = {}
    for k, v in config.items():
        if k in exclude_keys:
            continue
        
        if isinstance(v, dict) and "value" in v:
            sampled[k] = v["value"]
        elif isinstance(v, dict) and "min" in v and "max" in v:
            sampled[k] = np.random.uniform(v["min"], v["max"])
        elif isinstance(v, dict) and "values" in v:
            sampled[k] = np.random.choice(v["values"])
        else:
            # fallback: use raw value if not dict
            sampled[k] = v

    return sampled


def run_unifloral(config_dir, entity, project, run_limit=None, parameter_seed=42):
    """
        Loads configs, runs the experiments specified by config dataset-name
        and seed(s).

        entity, project specify the wandb entity and project to use.

        run_limit specifies an upper bound on the number of runs per env


        parameter seed controls the sampling of parameters, along with
        environment seeds supplied via the configs makes training
        deterministic*.

        * Antmaze does not support seeding, hence the asterisk.
    """

    loaded_configs = load_configs(config_dir)

    if parameter_seed is not None:
        np.random.seed(parameter_seed)

    for config in loaded_configs:

        if "parameters" not in config:
            raise ValueError("Sweep config must have 'parameters' section")

        if "dataset-name" not in config["parameters"]:
            raise ValueError("Sweep config must have 'dataset-name' parameter")

        # Get list of datasets
        datasets = get_param(config, "dataset-name")
        if not isinstance(datasets, list):
            datasets = [datasets]

        # Get list of seeds
        seeds = get_param(config, "seed")
        if not isinstance(seeds, list):
            seeds = [seeds]

        for env in datasets:

            print(f"Running experiments for environment: {env}")
            env_runs = 0

            for seed in seeds:

                if run_limit is not None and env_runs >= run_limit:
                    break

                # Sample parameters for this run
                sampled_params = sample_params(config["parameters"], exclude_keys=["seed", "dataset-name"])

                # Build new config for this run
                new_config = config.copy()
                for k, v in sampled_params.items():
                    new_config["parameters"][k] = {"value": v}
                new_config["parameters"]["dataset-name"] = {"value": env}
                new_config["parameters"]["seed"] = {"value": seed}

                # Initialize W&B run
                run = wandb.init(
                    entity=entity,
                    project=project,
                    config=new_config,
                    group=env,
                    reinit=True
                )

                run.finish()


def run_experiment(entity, project, folder, prior=False):
    sweep_folder(folder, entity, project)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run square reach experiment.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--experiment", type=str, required=True, help="Path to configs")


    args = parser.parse_args()
    run_experiment(args.entity, args.project, args.experiment)



