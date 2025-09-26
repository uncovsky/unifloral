import gymnasium as gym
import matplotlib.pyplot as plt

import sys
import time
import numpy as np
import os
from datetime import datetime
from minari import DataCollector
import random

import torch
import tqdm
import stable_baselines3 as sb3

# Fix seed to 15 for a specific initial state 
seed = 15

def train_expert_policy(env_name='Pendulum-v1', total_timesteps=500000,
                        save_path='sac_expert'):

    env = gym.make(env_name)
    model = sb3.SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    env.close()
    return model



def get_q_values(model, env, num_points=1000, seed=0):

    # Create a grid of actions
    low, high = env.action_space.low, env.action_space.high
    actions = np.linspace(low, high, num_points)

    # Get starting state
    state = env.reset(seed=seed)[0]

    # move everything to cpu
    actions_tensor = torch.tensor(actions, dtype=torch.float32).cpu()
    state_tensor = torch.tensor(state,
                                dtype=torch.float32).unsqueeze(0).repeat(num_points, 1).cpu()

    # Compute Q-values for each state-action pair
    # move model to cpu
    model.critic.to('cpu')
    q_values = model.critic(state_tensor, actions_tensor)

    # reduce two critics to one via minimum
    q_values = torch.min(q_values[0], q_values[1]).detach().numpy()
    # sample actions from actor

    sampled_actions = [ model.predict(state, deterministic=False)[0].item() for _ in range(1000)]
   

    np.savez("pendulum_q_values.npz", 
             actions=actions,
             q_values=q_values,
             sampled_actions=sampled_actions)

    return actions, q_values, sampled_actions

def plot_q_values(actions, q_values, sampled_actions):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(actions, q_values, label='Q-values', color='blue')
    ax1.set_xlabel('Action (Theta dot)')
    ax1.set_ylabel('Q-value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    ax1.set_title('Q-values with Sampled Actions Histogram')

    ax2 = ax1.twinx()

    counts, bins, patches = ax2.hist(
        sampled_actions,
        bins=30,
        density=True,
        alpha=0.5,
        color='orange',
        label='Sampled Actions',
    )

    q_min, q_max = np.min(q_values), np.max(q_values)
    scale_factor = 0.5 * (q_max - q_min) / counts.max()
    for patch in patches:
        patch.set_height(patch.get_height() * scale_factor)

    ax2.set_ylim(0, 0.5 * (q_max - q_min))
    ax2.axis('off')

    ax1.legend(loc='upper left')

    return fig, ax1, ax2



def collect_pendulum_data(env_name='Pendulum-v1', model_path='sac_expert', num_episodes=1000,
                        seed=seed  ):

    env = gym.make(env_name)
    model = sb3.SAC.load(model_path)

    collector = DataCollector(env)

    for episode in tqdm.tqdm(range(num_episodes), desc="Collecting episodes"):

        # We always start from the same state to demonstrate 
        # the erroneous extrapolation
        obs, info = collector.reset(seed=seed)
        done = False
        episode_reward = 0

        while not done:
            # We sample random actions from actor though
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = collector.step(action)
            done = terminated or truncated
            episode_reward += reward

    # Create minari dataset
    collector.create_dataset(
            dataset_id='pendulum_expert-v0',
            eval_env=env,
            ref_min_score=-200,
            ref_max_score=0,
            algorithm_name='SAC',
            author='uncovsky',
            description='Pendulum expert dataset collected using SAC'
    )

    env.close()

if __name__ == "__main__":
    # Train expert policy

    #expert_model = train_expert_policy(save_path="sac_expert")
    model = sb3.SAC.load("sac_expert")

    # Visualize Q-values
    env = gym.make('Pendulum-v1')

    actions, q_values, sampled_actions = get_q_values(model, env, seed=seed)
    fig, ax1, ax2 = plot_q_values(actions, q_values, sampled_actions)
    plt.show()

    # Collect dataset
    collect_pendulum_data(num_episodes=5000)


