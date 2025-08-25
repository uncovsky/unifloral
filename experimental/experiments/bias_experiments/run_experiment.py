from helper_scripts.experiment_utils import sweep_folder
import os

# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

DEFAULT_FOLDERS = {
    "square-reach": "experiments/bias_experiments/configs/square-reach",
    "pendulum": "experiments/bias_experiments/configs/pendulum",
}


def run_square_experiment(entity, project):
    sweep_folder(DEFAULT_FOLDERS["square-reach"], entity, project)
    

def run_pendulum_experiment(entity, project):
    sweep_folder(DEFAULT_FOLDERS["pendulum"], entity, project)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run square reach experiment.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--experiment", type=str, choices=["square-reach",
                                                           "pendulum"],
                        default="square-reach", help="Which experiment to run")

    args = parser.parse_args()


    if args.experiment == "square-reach":
        run_square_experiment(args.entity, args.project)
    elif args.experiment == "pendulum":
        run_pendulum_experiment(args.entity, args.project)

    else:
        raise ValueError(f"Unknown experiment {args.experiment}")
    




