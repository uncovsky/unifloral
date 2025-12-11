import yaml
import wandb
import time

from helper_scripts.experiment_utils import load_configs

def get_param(config, key, default=None):
    """Helper to fetch parameter values from the sweep config."""
    return config.get("parameters", {}).get(key, {}).get("value", default)

def wandb_sweep_from_config(config, 
                            run_limit=None, 
                            environment=None,
                            project=None,
                            entity=None):
    """
    Initializes wandb.sweep from provided config
    Runs wandb agent & executes the sweep (blocking call)
    """
    assert "program" in config, "Sweep config must contain 'program' key"
    assert "parameters" in config, "Sweep config must contain 'parameters' key"
    assert "dataset-name" in config['parameters'], "Sweep config must contain 'dataset-name' parameter"

    config["parameters"]["eval_final_episodes"]["value"] = 0
    config["parameters"]["eval_interval"]["value"] = 1000
    config["parameters"]["num_updates"]["value"] = 10_000
    config["parameters"]["dataset-name"] = {"value": "hopper-medium-v2"}
    config["project"] = "comp_time_test"

    algo_name = config.get("name", "unspecified_algo")
    steps = get_param(config, "num_updates")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if entity is not None:
        config["entity"] = entity

    if environment is not None:
        print(f"Replacing config environment with {environment}.")
        config["parameters"]["dataset-name"] = {"value": environment}

    config["name"] = f"{algo_name}-{timestamp}"
    sweep_id = wandb.sweep(config)
    print(f"Sweep created for {algo_name} with id: {sweep_id}, steps: {steps}")

    # blocks until agent terminates
    wandb.agent(sweep_id, function=None, count=run_limit)

    config["name"] = algo_name  # reset name


if __name__ == "__main__":

    cfg = load_configs("experiments/unifloral_eval/comp_time")
    for config in cfg:
        wandb_sweep_from_config(
            config,
            run_limit=1
        )
    

