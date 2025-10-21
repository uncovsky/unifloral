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


def collect_d_dim_data(episodes=1000, ds=[5,10,20], seeds=[0,1,2]):
    for i, d in enumerate(ds):
        seed = seeds[i % len(seeds)]
        np.random.seed(seed)
        env = gym.make("DDimensionalBandit-v0", d=d, epsilon=0.3)
        collector = DataCollector(env)
        obs, _ = collector.reset(seed=seed)


        
        def sample_from_epsilon_ball(d, epsilon):
            # random direction on the sphere
            x = np.random.normal(0, 1, size=d)
            x /= np.linalg.norm(x)

            # random radius with correct distribution for uniform ball
            r = epsilon * np.random.rand() ** (1.0 / d)
            return (r * x).astype(np.float32)

        for ep in tqdm.tqdm(range(episodes), desc=f"Collecting {d}D Bandit expert data, seed {seed}"):
            done = False

            while not done:
                a = sample_from_epsilon_ball(d, epsilon=0.3)
                next_obs, reward, terminated, truncated, _ = collector.step(a)
                print(a, reward)
                done = terminated or truncated
                obs = next_obs

            # same state every ep

        # Create minari dataset
        collector.create_dataset(
                dataset_id=f'bandit_{d}-v0',
                eval_env=env,
                ref_min_score=-1,
                ref_max_score=1,
                algorithm_name='expert',
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


def collect_uniform_data(episodes=10000, seed=0):

    np.random.seed(seed)
    env = gym.make("ContinuousBandit-v0")
    collector = DataCollector(env)

    obs, _ = collector.reset(seed=seed)

    for ep in tqdm.tqdm(range(episodes), desc="Collecting Bandit expert data"):
        a = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
        next_obs, reward, terminated, truncated, _ = collector.step(a)
        obs, _ = collector.reset()
        # same state every ep

    # Create minari dataset
    collector.create_dataset(
            dataset_id='bandit_uniform-v0',
            eval_env=env,
            ref_min_score=-1,
            ref_max_score=1,
            algorithm_name='SAC',
            author='uncovsky',
    )
        




