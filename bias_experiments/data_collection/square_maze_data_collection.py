import gymnasium as gym
from gymnasium import spaces
from minari import DataCollector, load_dataset

import mock_environments
import matplotlib.pyplot as plt
import numpy as np

def collect_dataset(H, episodes=1000, seed=0):

    env = gym.make("SquareReachEnv-v0", H=H, render_mode="human")

    # collect data
    collecting_env = DataCollector(env)

    rng = np.random.default_rng(seed)
    dataset = []

    for ep in range(episodes):
        obs, _ = collecting_env.reset()
        done = False
        t = 0

        # act randomly for first two steps, then deterministically navigate to goal
        random_steps = min(H // 5, 5)

        while not done:
            if t < random_steps:
                # Random heading in (-1, 1)
                a = np.array([rng.uniform(-1, 1)], dtype=np.float32)
            else:
                # Aim at goal
                gx, gy = 1.0, 1.0
                vx, vy = gx - obs[0], gy - obs[1]
                theta = np.arctan2(vy, vx)          # radians (-pi, pi)
                a = np.array([np.clip(theta / np.pi, -1.0, 1.0)], dtype=np.float32)

            next_obs, reward, terminated, truncated, _ = collecting_env.step(a)
            dataset.append((obs, a, reward, next_obs, terminated or truncated))

            obs = next_obs
            done = terminated or truncated
            t += 1

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
