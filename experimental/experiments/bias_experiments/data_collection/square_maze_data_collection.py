import gymnasium as gym
from gymnasium import spaces
from minari import DataCollector, load_dataset

import mock_environments
import matplotlib.pyplot as plt
import numpy as np

def collect_dataset(H, episodes=5000, seed=0):

    
    env = gym.make("SquareReachEnv-v0", H=H, render_mode="human")

    # collect data
    collecting_env = DataCollector(env)

    """
        Phase one - sample random initial states, navigate to goal
    """
    # Seed the rng for initial state sampling, sample random state
    env.unwrapped.seed_init(seed)
    # Enable sampling of initial states
    env.unwrapped.set_randomize(True)

    phase_episodes = episodes // 2
    obs, _ = collecting_env.reset(seed=seed)

    for ep in range(phase_episodes):
        done = False
        while not done:
            # navigate to goal deterministically
            gx, gy = 1.0, 1.0
            vx, vy = gx - obs[0], gy - obs[1]
            theta = np.arctan2(vy, vx)          # radians (-pi, pi)
            a = np.array([np.clip(theta / np.pi, -1.0, 1.0)], dtype=np.float32)

            next_obs, reward, terminated, truncated, _ = collecting_env.step(a)

            obs = next_obs
            done = terminated or truncated

            if done:
                obs, _ = collecting_env.reset()



    """
        Phase two - fixed initial state, random actions
    """ 

    env.unwrapped.set_randomize(False)
    rng = np.random.default_rng(seed)
    obs, _ = collecting_env.reset()

    random_steps = H // 5

    for ep in range(phase_episodes):
        done = False
        t = 0
        while not done:
            if t < random_steps:
                a = np.array([rng.uniform(-1, 1)], dtype=np.float32)

            else:
                # navigate to goal deterministically
                gx, gy = 1.0, 1.0
                vx, vy = gx - obs[0], gy - obs[1]
                theta = np.arctan2(vy, vx)
                a = np.array([np.clip(theta / np.pi, -1.0, 1.0)], dtype=np.float32)

            t += 1

            next_obs, reward, terminated, truncated, _ = collecting_env.step(a)

            obs = next_obs
            done = terminated or truncated

            if done:
                obs, _ = collecting_env.reset()


    

    # save plot of first 10 trajectories
    env.unwrapped.plot_trajectories()

    dataset_id = f"square-reach/horizon-{H}-v0"

    collecting_env.create_dataset(
        dataset_id=dataset_id,
        eval_env=env,
        ref_min_score=0.0,
        ref_max_score=1.0,
        algorithm_name="uniform",
        author="uncovsky",
        description=f"Simple goal reaching env to test reward propagation, horizon {H}.",
    )

    env.close()

collect_dataset(H=22, steps=100_000, seed=0)
