from collections import namedtuple
from datetime import datetime
import gymnasium as gym
import jax
import jax.numpy as jnp
import json

from flax.linen.initializers import constant, uniform
from flax.training import checkpoints
from flax import linen as nn

import matplotlib.pyplot as plt
import minari
import numpy as np
import random
import seaborn as sns
import os


# algorithm-specific hyperparameters
cql_params = ['cql_temperature', 'cql_min_q_weight']
sac_params = ['num_critics']
msg_params = ['num_critics', 'cql_min_q_weight', 'actor_lcb_coef']

hyperparams = {}

hyperparams["cql"] = cql_params
hyperparams["sac_n"] = sac_params
hyperparams["msg"] = msg_params




"""
    Policies used for data collection, see
    experiments/bias_experiments/data_collection/pendulum_data_collection.py
"""
def gaussian_mixture_policy(_):

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

def uniform_mixture_policy(_):
    """
        Mixture of two uniform distributions [-2, -1] and [1, 2]
    """
    coin = random.uniform(0, 1)
    if coin < 0.5:
        res = np.random.uniform(low=-2, high=-1)
    else:
        res = np.random.uniform(low=1, high=2)
    return np.array([res], dtype=np.float32)


def uniform_policy(_):
    # just sample uniformly, turn to array
    res = np.random.uniform(low=-2, high=2)
    return np.array([res], dtype=np.float32)  

"""
    Module forward for ensemble Q-function
"""
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



def get_q_value_data(directory, num_samples):

    # Get results for different initial policies
    policy_folders = [f.path for f in os.scandir(directory) if f.is_dir()]

    initial_obs = gym.make("Pendulum-v1").reset(seed=42)[0]
    action_samples = np.linspace(-2, 2, num_samples).reshape(-1, 1).astype(np.float32)
    obs = jnp.repeat(jnp.array(initial_obs).reshape(1, -1), num_samples, axis=0)

    for policy_folder in policy_folders:
        agent_folders = [f.path for f in os.scandir(policy_folder) if f.is_dir()]
        policy_name = os.path.basename(policy_folder)
        results = {}

        for agent_folder in agent_folders:
            checkpoint_folders = [f.path for f in os.scandir(agent_folder) if f.is_dir()]

            for check_folder in checkpoint_folders:
                folders = [f.path for f in os.scandir(check_folder) if
                           f.is_dir()]

                last_checkpoint = folders[-1]
                args = json.load(open(os.path.join(check_folder,"args.json"), "r"))
                checkpoint = checkpoints.restore_checkpoint(last_checkpoint, target=None)
                vec_q_params = checkpoint['vec_q']['params']

                state_dim = 3
                action_dim = 1
                num_critics = args['num_critics']

                vec_q_module = VectorQ(num_critics=num_critics)
                dummy_obs = jnp.ones((1, state_dim))
                dummy_action = jnp.ones((1, action_dim))

                _ = vec_q_module.init(jax.random.PRNGKey(0), dummy_obs, dummy_action)

                q_values = vec_q_module.apply(vec_q_params, obs, action_samples)
                algo = args['algorithm']

                # Reduce Q-values based on algorithm

                if algo in ['sac_n', 'cql']:
                    q_values = q_values.min(axis=-1)

                else:
                    q_values_final = q_values.mean(axis=-1)
                    q_values_final += args['actor_lcb_coef'] * q_values.std(axis=-1)
                    q_values = q_values_final

                params = {k: args[k] for k in hyperparams[args['algorithm']] + ['seed', 'algorithm']}
                arg_str = "_".join([f"{k}-{v}" for k, v in params.items()])
                arg_str = f"{policy_name}_{arg_str}"
                
                results[arg_str] = q_values



        sample_policy = gaussian_mixture_policy if policy_name == "gaussian" else (
                        uniform_mixture_policy if policy_name == "mixture" else uniform_policy) 

        policy_samples = np.concatenate([ sample_policy(None) for _ in range(num_samples * 10)], axis=0)


        def overlay_kde_seaborn_center(ax, samples, frac=0.3, bw_adjust=0.5, linewidth=2.5, linestyle="--", color="black", label=None):
            """
            Overlay a smooth KDE using seaborn, anchored at the vertical midpoint of the y-axis.
            - frac: fraction of y-range to occupy
            - bw_adjust: bandwidth adjustment (smaller=bumpier, larger=smoother)
            """
            # Fit KDE with Seaborn
            kde = sns.kdeplot(samples, bw_adjust=bw_adjust, ax=ax, fill=False, legend=False)

            # Extract line data from the plot
            line = ax.lines[-1]
            xs = line.get_xdata()
            ys = line.get_ydata()

            # Remove the line added by seaborn
            line.remove()

            # Normalize and scale
            ymin, ymax = ax.get_ylim()
            ymid = 0.5 * (ymin + ymax)
            yrange = ymax - ymin
            scale = frac * yrange
            ys_scaled = ymid + ys / ys.max() * scale  # only extend upward from midpoint

            ax.plot(xs, ys_scaled, linewidth=linewidth, linestyle=linestyle, color=color, label=label, zorder=1)


        fig, ax = plt.subplots(figsize=(10, 6))

        for label, q_vals in results.items():
            ax.plot(action_samples, q_vals, label=label)
        overlay_kde_seaborn_center(ax, policy_samples,
                                   frac=0.3, bw_adjust=0.5,
                                   linewidth=2.5, linestyle="--",
                                   color="black",
                                   label="Policy KDE")

        ax.set_xlabel("Action")
        ax.set_ylabel("Q-value")
        ax.set_xlim(-2, 2)
        ax.legend()
        plt.show()



AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha pretrain_lag")

get_q_value_data("../results/updated_checkpoints_pendulum", 1000)


