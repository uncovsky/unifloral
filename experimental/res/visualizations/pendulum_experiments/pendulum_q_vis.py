import jax
import gymnasium as gym

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, uniform
from flax.training import checkpoints
import os


# Same seed for initial pendulum state as in data collection
seed = 15

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


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale
    return _init


class SoftQNetwork(nn.Module):
    depth: int = 3
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        q = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))(x)
        return q.squeeze(-1)

class VectorQ(nn.Module):
    num_critics: int
    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action)
        return q_values


def get_q_values_for_ckpt(ckpt_path='./checkpoint'):
    abs_path = os.path.abspath(ckpt_path)
    checkpoint = checkpoints.restore_checkpoint(abs_path, target=None)
    vec_q_params = checkpoint['vec_q']['params']

    """
        Fixed hyperparameters for simplicity
    """
    state_dim = 3
    action_dim = 1
    num_critics = 10

    vec_q_module = VectorQ(num_critics=num_critics)
    dummy_obs = jnp.ones((1, state_dim))

    dummy_action = jnp.ones((1, action_dim))

    _ = vec_q_module.init(jax.random.PRNGKey(0), dummy_obs, dummy_action)

    env = gym.make("Pendulum-v1")

    # Use same seed like in dataset collection.
    state, _ = env.reset(seed=seed)

    try:
        expert_data = np.load("pendulum_q_values.npz", allow_pickle=True)
    except:
        print("Run data collection & expert traiining first")
        return

    actions = expert_data['actions']
    q_values = expert_data['q_values']
    sampled_actions = expert_data['sampled_actions']
    fig, ax1, ax2 = plot_q_values(actions, q_values, sampled_actions)

    state_tensor = jnp.array(state, dtype=jnp.float32).reshape(1, -1)
    state_tensor = state_tensor.repeat(actions.shape[0], axis=0)
    actions_tensor = jnp.array(actions, dtype=jnp.float32).reshape(-1, 1)

    ensemble_q_values = vec_q_module.apply(vec_q_params, state_tensor, actions_tensor)
    # Reduce Q-vals over ensemble dimension
    ensemble_q_values = jnp.min(ensemble_q_values, axis=-1)

    ax1.plot(actions, ensemble_q_values, label='Ensemble Q-values', color='green')
    ax1.legend(loc='upper left')

    plt.show()

if __name__ == "__main__":
    get_q_values_for_ckpt('./pbrl_checkpoint')

