
import os
import sys
import signal
import time
import yaml
import wandb


# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main(config_path):
    with open(config_path, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    algorithm_name = sweep_cfg.get("name", "default-algorithm")

    print(f"Creating sweep for algorithm: {algorithm_name}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    sweep_cfg["name"] = f"{algorithm_name}-{timestamp}"

    sweep_id = wandb.sweep(sweep_cfg)

    print(f"Sweep created with id: {sweep_id}")
    wandb.agent(sweep_id, function=None)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python wandb_sweep_driver.py <sweep_config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
