import gymnasium as gym
import matplotlib.pyplot as plt

import sys
import time
import numpy as np
import os
from datetime import datetime
from minari import DataCollector
import random
from stable_baselines3 import PPO
from rl_zoo3.train import train


def gaussian_mixture_policy_acro(_):

    """
        GMM for Pendulum, other envs need to adjust clipping and dimensionality!
    """

    # mix of two gaussians with means -1 and 1, stddev 0.5
    # Flip coin for gaussian mixture
    coin = random.uniform(0, 1)
    mean = -1 if coin < 0.5 else 1
    stddev = 0.2

    # Sample from the gaussian
    act = np.random.normal(loc=mean, scale=stddev)

    act = np.clip(act, -2, 2)  # Clip to action space limits
    return np.array([act], dtype=np.float32)

def uniform_mixture_policy_acro(_):
    """
        Mixture of two uniform distributions [-2, -1] and [1, 2]
    """
    coin = random.uniform(0, 1)
    if coin < 0.5:
        res = np.random.uniform(low=-2, high=-1)
    else:
        res = np.random.uniform(low=1, high=2)
    return np.array([res], dtype=np.float32)


def uniform_policy_acro(_):
    # just sample uniformly, turn to array
    res = np.random.uniform(low=-2, high=2)
    return np.array([res], dtype=np.float32)  


def collect_expert_dataset(env, model_path, dataset_size, seed=42):

    """
        Collects a minari dataset on `env` using `expert_policy` in
        initial state, and random actions thereafter.
    """

    agent = PPO.load(model_path)

    step = 0

    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)



    obs, _ = env.reset()

    while step < dataset_size:
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if terminated or truncated:
            obs, _ = env.reset()

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset_id = f"pendulum/ppo-expert-v1"
    algorithm_name = "Expert"
    author = "uncovsky"

    env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name=algorithm_name,
        author=author,
        description=f"Collected by trained PPO agent on Pendulum-v1, seed {seed}.",
    )


def collect_uniform_dataset(env, first_state_policy, dataset_size, seed=42):

    """
        Collects a minari dataset on `env` using `first_state_policy` in
        initial state, and random actions thereafter.
    """

    step = 0
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    action = first_state_policy(obs)

    while step < dataset_size:
        obs, reward, terminated, truncated, info = env.step(action)
        action = env.action_space.sample()  # Random action after first state
        step += 1

        if terminated or truncated:
            obs, _ = env.reset()
            action = first_state_policy(obs)


    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset_id = f"pendulum/gauss-mixed-v2"
    print(dataset_id)
    algorithm_name = "Uniform"
    author = "uncovsky"

    env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name=algorithm_name,
        author=author,
        description=f"Uniformly collected dataset on Pendulum-v1, seed {seed}.",
    )



def visualize_policy_histogram(policy, env, num_samples=10000):
    """
    Visualizes the histogram of actions taken by a given policy in the environment.
    """
    actions = [policy(env.reset()[0])[0] for _ in range(num_samples)]
    plt.hist(actions, bins=50, density=True)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Density")
    plt.grid()
    plt.show()






if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    collecting_env = DataCollector(gym.make("Pendulum-v1"))


    """
    collect_uniform_dataset(
        collecting_env, gaussian_mixture_policy_acro, dataset_size=100000, seed=42
    )

    # Train agent
    sys.argv = ["python", "--algo", "ppo", "--env", "Pendulum-v1", "--n-timesteps", "1000000", "--seed", "42"]
    train()


    visualize_policy_histogram(gaussian_mixture_policy_acro, env, num_samples=10000)
    visualize_policy_histogram(uniform_mixture_policy_acro, env,
                               num_samples=10000)
    visualize_policy_histogram(uniform_policy_acro, env, num_samples=10000)



    """

    path = os.path.abspath('') + '/logs/ppo/Pendulum-v1_1/best_model'
    collect_expert_dataset(collecting_env, path, dataset_size=100000, seed=42)




