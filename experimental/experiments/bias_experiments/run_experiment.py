from helper_scripts.experiment_utils import load_configs, wandb_sweep_from_config

import os

# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

square_cfg_dir = "experiments/bias_experiments/configs/square-reach"

def run_square_experiment(entity, project):
    """
    Run the square reach experiment using the specified configuration.
    """
    
    configs = load_configs(square_cfg_dir)
    for config in configs:
        wandb_sweep_from_config(
            config,
            project=project,
            entity=entity
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run square reach experiment.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")

    args = parser.parse_args()

    run_square_experiment(args.entity, args.project)




