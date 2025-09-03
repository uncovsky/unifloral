import gymnasium as gym
from gymnasium import spaces
from minari import DataCollector, load_dataset

import mock_environments
import matplotlib.pyplot as plt
import numpy as np

def collect_dataset(H, steps=100000, seed=0):
    np.random.seed(seed)
    
    env = gym.make("SquareReachEnv-v0", H=H, render_mode="human")

    # collect data
    collecting_env = DataCollector(env)
    step = 0
    env.unwrapped.set_randomize(False)

    while step < steps:
        done = False
        obs, _ = collecting_env.reset()

        coin = np.random.uniform()
        a = np.array([0.0], dtype=np.float32) if coin < 0.5 else np.array([0.25], dtype=np.float32)

        while not done:
            step += 1
            next_obs, reward, terminated, truncated, _ = collecting_env.step(a)

            obs = next_obs
            done = terminated or truncated

            if done:
                obs, _ = collecting_env.reset()


    
    dataset_id = f"bc-fail/horizon-{H}-v0"
    collecting_env.create_dataset(
        dataset_id=dataset_id,
        eval_env=env,
        ref_min_score=0.0,
        ref_max_score=1.0,
        algorithm_name="mixed",
        author="uncovsky",
        description=f"Simple dataset that breaks TD3+bc, horizon {H}.",
    )

if __name__ == "__main__":
    collect_dataset(H=50, steps=100000, seed=0)

