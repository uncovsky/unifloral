from helper_scripts.experiment_utils import sweep_folder
import os

# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

folder = "experiments/bc_fail_experiment/configs"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run square reach experiment.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--experiment", type=str, choices=["square-reach",
                                                           "pendulum"],
                        default="square-reach", help="Which experiment to run")

    args = parser.parse_args()


    sweep_folder(folder, args.entity, args.project)

    




