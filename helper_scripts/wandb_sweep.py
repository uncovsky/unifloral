
import os
import sys
import signal
import time
import yaml
import wandb


# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

MUJOCO_TASKS = [
  "halfcheetah-random-v2",
  "halfcheetah-medium-v2",
  "halfcheetah-expert-v2",
  "halfcheetah-medium-expert-v2",
  "halfcheetah-medium-replay-v2",
  "hopper-random-v2",
  "hopper-medium-v2",
  "hopper-expert-v2",
  "hopper-medium-expert-v2",
  "hopper-medium-replay-v2",
  "walker2d-random-v2",
  "walker2d-medium-v2",
  "walker2d-expert-v2",
  "walker2d-medium-expert-v2",
  "walker2d-medium-replay-v2",
]


ANT_TASKS = [
    #"antmaze-umaze-diverse-v2",
    #"antmaze-umaze-v2",
    #antmaze-medium-play-v2",
    #antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
]



def create_sweep_for_env(base_sweep_config, env_name, project_name, entity_name):
    sweep_cfg = base_sweep_config.copy()

    # Fix environment param value, remove sampling if exists
    if "parameters" not in sweep_cfg:
        raise ValueError("Sweep config must have 'parameters' section")

    if "dataset" not in sweep_cfg["parameters"]:
        raise ValueError("Sweep config must have 'dataset' parameter")

    sweep_cfg["parameters"]["dataset"] = {
        "value": env_name
    }

    sweep_cfg["project"] = project_name
    sweep_cfg["entity"] = entity_name

    algorithm_name = sweep_cfg.get("name", "default-algorithm")

    print(f"Creating sweep for env: {env_name}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    sweep_cfg["name"] = f"{algorithm_name}-{env_name}-{timestamp}"

    sweep_id = wandb.sweep(sweep_cfg)
    print(f"Sweep created with id: {sweep_id}")
    return sweep_id


def main(config_path):
    with open(config_path, "r") as f:
        base_sweep_config = yaml.safe_load(f)

    algorithm_name = sweep_cfg.get("name", "default-algorithm")

    print(f"Creating sweep for algorithm: {algorithm_name}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    sweep_cfg["name"] = f"{algorithm_name}-{env_name}-{timestamp}"

    sweep_id = wandb.sweep(sweep_cfg)

    print(f"Sweep created with id: {sweep_id}")
    return sweep_id

    wandb.agent(sweep_id, function=None)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python wandb_sweep_driver.py <sweep_config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
