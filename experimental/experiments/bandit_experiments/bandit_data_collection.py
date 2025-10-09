import numpy as np
import gymnasium as gym
import mock_environments
from minari import DataCollector
import tqdm

"""
    Data collection for the coninuous Bandit benchmark.

        see mock_environments/continuous_bandit.py
"""

def mixed_expert_policy(_):
    """
        Bandit has [-1, 1] action space

        With optimal actions 
            [-1, -0.5] u [0.5, 1] -> reward +1
            otherwise rew -1

            we play mixed optimal actions
    """
    if np.random.rand() < 0.5:
        return np.random.uniform(-1, -0.5, size=(1,)).astype(np.float32)
    else:
        return np.random.uniform(0.5, 1, size=(1,)).astype(np.float32)

def collect_bandit_data(episodes=1000, seed=0):

    np.random.seed(seed)
    env = gym.make("ContinuousBandit-v0")
    collector = DataCollector(env)
    policy = mixed_expert_policy

    obs, _ = collector.reset(seed=seed)

    for ep in tqdm.tqdm(range(episodes), desc="Collecting Bandit expert data"):
        done = False

        while not done:
            a = policy(obs)
            next_obs, reward, terminated, truncated, _ = collector.step(a)
            done = terminated or truncated
            obs = next_obs

        # same state every ep

    # Create minari dataset
    collector.create_dataset(
            dataset_id='bandit_expert-v0',
            eval_env=env,
            ref_min_score=-1,
            ref_max_score=1,
            algorithm_name='SAC',
            author='uncovsky',
    )

def collect_cql_data(episodes=10000, seed=0):

    np.random.seed(seed)
    env = gym.make("ContinuousBandit-v0")
    collector = DataCollector(env)
    policy = mixed_expert_policy

    obs, _ = collector.reset(seed=seed)

    for ep in tqdm.tqdm(range(episodes), desc="Collecting Bandit expert data"):

        if ep > 9/10 * episodes:
            a = policy(obs)  # expert in last 10% of data
        else:
            # 90% of data is random
            a = np.random.uniform(-0.5, 0.5, size=(1,)).astype(np.float32)

        next_obs, reward, terminated, truncated, _ = collector.step(a)
        obs, _ = collector.reset()


        # same state every ep

    # Create minari dataset
    collector.create_dataset(
            dataset_id='bandit_cql-v0',
            eval_env=env,
            ref_min_score=-1,
            ref_max_score=1,
            algorithm_name='SAC',
            author='uncovsky',
    )
