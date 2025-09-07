from helper_scripts.experiment_utils import sweep_folder
import os

# prevent jax from prealloc mem
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

folder = "experiments/diversity_experiments/configs"
folder_priors = "experiments/diversity_experiments/configs/priors"

def run_experiment(entity, project, prior=False):
    if prior:
        sweep_folder(folder_priors, entity, project)
    else:
    s   weep_folder(folder, entity, project)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run square reach experiment.")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--prior", type=bool, default=False, help="Whether to use randomized priors")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")

    # help
    args = parser.parse_args()
    run_experiment(args.entity, args.project, args.prior)



