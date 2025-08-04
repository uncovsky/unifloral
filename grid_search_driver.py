"""
    This script manages algorithm training, utilizing the existing CLI interface
    for the algorithms. Specifies the search space for all the algorithms that
    were trained, along with the datasets, etc.
"""

import os
import sys

# prevent from grabbing all mem by XLA backend
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax

import d4rl
import time
import gym
import jax.numpy as jnp
import subprocess


MSG_PARAM_BOX = {
        "cql_min_q_weight" :  [0.0, 0.5],
        "actor_lcb_coef" : [1.0, 8.0],
}


# Make the scripts run with gpu
algo_env = os.environ.copy()
algo_env.pop("JAX_PLATFORM_NAME", None)


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


def sample_hyperparameters(hyperparam_box, rng):
    """
    Sample hyperparameters for the algorithm.
    """
    rng_grid, rng = jax.random.split(rng)

    # Get number of hyperparameters to sample
    rng_params = jax.random.split(rng_grid, len(hyperparam_box.keys()))

    hyperparams = {}
    for i, (key, value) in enumerate(hyperparam_box.items()):
        low, high = value
        hyperparams[key] = jax.random.uniform(rng_params[i], minval=low, maxval=high)

    return hyperparams





def train():

    """
    Main function to train the algorithm.
    """

    algorithms = ["msg"]

    # Do locomotion
    datasets = MUJOCO_TASKS

    wandb_team = "ahoj"


    learning_steps = 1000000
    training_seeds_num = 5

    # Initialize random number generator
    rng = jax.random.PRNGKey(42)

    # product of algorithms and datasets
    for algorithm in algorithms:
        for dataset in datasets:

            timestamp = time.strftime("%Y%m%d-%H%M%S")

            rng_seed, rng = jax.random.split(rng)
            rng_seeds = jax.random.split(rng_seed, training_seeds_num)

            def _rng_to_integer_seed(rng):
                return int(jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max))

            seeds = [_rng_to_integer_seed(seed) for seed in rng_seeds]

            for seed in seeds:
                # Sample hyperparameters
                rng, rng_hyperparams = jax.random.split(rng)
                hyperparams = sample_hyperparameters(MSG_PARAM_BOX, rng_hyperparams)
                print(f"Training {algorithm} on {dataset} with seed {seed} with hyperparameters {hyperparams}")

                command = [
                    "nice", "-n", "19",
                    "python3", f"algorithms/{algorithm}.py",
                    "--log",
                    "--dataset", dataset,
                    "--num_updates", str(learning_steps),
                    "--wandb_team", wandb_team,
                    "--wandb_project", f"{algorithm}_{dataset}_{timestamp}",
                    "--seed", str(seed),
                ]

                # unwrap hyperparameters
                for key, value in hyperparams.items():
                    command.extend([f"--{key}", f"{value:.4f}"])

                print("Running:", command)
                subprocess.run(command, check=True, env=algo_env)
                    

if __name__ == "__main__":
    train()


