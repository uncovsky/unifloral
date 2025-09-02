from helper_scripts.experiment_utils import sweep_folder
import os

# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

folder = "experiments/diversity_experiments/configs"

def run_experiment(entity, project):
    sweep_folder(folder, entity, project)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run square reach experiment.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")

    args = parser.parse_args()
    run_experiment(args.entity, args.project)



