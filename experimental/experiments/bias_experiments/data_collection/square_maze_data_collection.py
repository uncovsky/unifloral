import gymnasium as gym
from gymnasium import spaces
from minari import DataCollector, load_dataset

import mock_environments
import matplotlib.pyplot as plt
import numpy as np

def action_to_goal(obs, gx, gy):
    # Get action to navigate to [gx,gy]
    vx, vy = gx - obs[0], gy - obs[1]
    theta = np.arctan2(vy, vx)
    a = np.array([np.clip(theta / np.pi, -1.0, 1.0)], dtype=np.float32)
    return a


def dist(obs, gx, gy):
    return np.sqrt((obs[0] - gx)**2 + (obs[1] - gy)**2)


def collect_dataset(H, noise_eps, 
                       episodes=1000,
                       seed=0):

    """
        Collects a dataset on square reach:

        Phase one - sample random waypoints, navigate back to start
                        (confounding trajectories)

        Phase two - sample random waypoint, navigate to goal
    
    """

    env = gym.make("SquareReachEnv-v0", H=H, render_mode="human")

    # collect data
    collecting_env = DataCollector(env)

    """
        Phase one
    """
    # Seed the rng for initial state sampling, but turn it off
    env.unwrapped.set_randomize(False)
    env.unwrapped.seed_init(seed)
    np.random.seed(seed)

    # get step size
    step_size = env.unwrapped.step_size


    for ep in range(episodes):

        obs, _ = collecting_env.reset()
        done = False

        while not done:
            a = action_to_goal(obs, 1.0, 1.0)
            a += np.random.normal(0, noise_eps, size=a.shape)
            np.clip(a, -1.0, 1.0, out=a)
            next_obs, reward, terminated, truncated, _ = collecting_env.step(a)
            obs = next_obs
            done = terminated or truncated


    env.unwrapped.plot_trajectories()

    noise_eps = int(noise_eps*10)

    dataset_id = f"square-reach/horizon-{H}-eps{noise_eps}-v0"

    collecting_env.create_dataset(
        dataset_id=dataset_id,
        eval_env=env,
        ref_min_score=0.0,
        ref_max_score=1.0,
        algorithm_name="uniform",
        author="uncovsky",
        description=f"Simple goal reaching env to test reward propagation, horizon {H}, noise={noise_eps}",)

    env.close()
