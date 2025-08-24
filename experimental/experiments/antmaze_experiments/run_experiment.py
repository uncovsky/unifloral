from helper_scripts.experiment_utils import load_configs, wandb_sweep_from_config

import os

# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

square_cfg_dir = "experiments/antmaze_experiments/configs/"

def run_antmaze_experiment(entity, project):
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

    parser = argparse.ArgumentParser(description="Run antmaze pretraining experiments.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")

    args = parser.parse_args()

    run_antmaze_experiment(args.entity, args.project)




