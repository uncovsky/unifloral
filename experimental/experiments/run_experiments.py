import argparse
import os
import sys
import signal
import time
import yaml
import wandb

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
    
def run_bias_experiments(config_dir, run_limit=None):
    print("Starting bias experiments...")

    loaded_configs = load_configs(config_dir)

    for config in loaded_configs:
        run_sweep(config, run_limit)


def run_antmaze_experiments(config_dir, run_limit=None):
    print("Starting AntMaze experiments...")

    loaded_configs = load_configs(config_dir)

    for env_name in ANT_TASKS:
        for config in loaded_configs:

            if "parameters" not in config:
                raise ValueError("Sweep config must have 'parameters' section")
            if "dataset" not in sweep_cfg["parameters"]:
                raise ValueError("Sweep config must have 'dataset' parameter")

            config["parameters"]["dataset"] = {
                "value": env_name
            }

            run_sweep(config, run_limit)


def run_mujoco_experiments(config_dir, run_limit=None):
    print("Starting MuJoCo experiments...")

    loaded_configs = load_configs(config_dir)

    for env_name in MUJOCO_TASKS:
        for config in loaded_configs:

            if "parameters" not in config:
                raise ValueError("Sweep config must have 'parameters' section")
            if "dataset" not in config["parameters"]:
                raise ValueError("Sweep config must have 'dataset' parameter")

            config["parameters"]["dataset"] = {
                "value": env_name
            }

            run_sweep(config, run_limit)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run bias experiments with specified configurations.")
    arg_parser.add_argument("--run_limit", type=int, default=None)
    arg_parser.add_argument("--config_dir", type=str, default=bias_exp_config_path,
                            help="Directory containing configuration files for bias experiments.")
    args = arg_parser.parse_args()

    run_limit = args.run_limit
    config_dir = args.config_dir

    print(f"Config directory: {config_dir}, Run limit: {run_limit}")
    if not os.path.exists(config_dir):
        print(f"Configuration directory {config_dir} does not exist.")
        sys.exit(1)

    run_bias_experiments(config_dir, run_limit)
