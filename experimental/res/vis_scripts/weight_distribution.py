import distrax
import numpy as np
import jax
import jax.numpy as jnp
import json
import os

from flax.training import checkpoints
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from functools import partial

from scipy.stats import gaussian_kde
from tex_setup import set_size

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


""" 

   Loads checkpoints of models
   and visualizes the weight distributions
   of first layer of the critics

"""

def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale
    return _init

"""
    Critic model
"""
class SoftQNetwork(nn.Module):
    depth: int = 3
    critic_norm: str = "none"
    learnable: bool = True
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)

        for _ in range(self.depth):
            layer = nn.Dense(256, bias_init=constant(0.1))

            if self.learnable:
                # Normalization for learnable Q-nets
                if self.critic_norm == "layer":
                    x = layer(x)
                    x = nn.LayerNorm()(x)
                else:
                    x = layer(x)
            else:
                x = layer(x)

            x = nn.relu(x)

        if self.learnable:
            last_layer = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))
        else:
            last_layer = nn.Dense(1, kernel_init=he_normal(), bias_init=sym(3e-3))
        q = last_layer(x)

        return q.squeeze(-1)

class VectorQ(nn.Module):
    num_critics: int
    depth : int
    critic_norm: str = "none"
    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
                partial(SoftQNetwork, critic_norm=self.critic_norm,
                                      depth=self.depth,
                                      learnable=True), 
                variable_axes={"params": 0, "batch_stats" : 0},  
                split_rngs={"params": True, "dropout": True},  
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action)
        return q_values


def collect_folders(ckpt_dirs):
    """Collect all folder names to assign stable colors."""
    folders = []
    for d in ckpt_dirs:
        for f in os.listdir(d):
            folders.append(f)
    return sorted(folders)


def load_checkpoints(ckpt_dir):
    folders = sorted(os.listdir(ckpt_dir))

    weights_dict = {}

    for i, folder in enumerate(sorted(folders)):
        color = cmap(i % 10)         

        for checkpoint in os.listdir(os.path.join(ckpt_dir, folder)):
            checkpoint_folder = os.path.join(ckpt_dir, folder, checkpoint)

            args = json.load(open(os.path.join(checkpoint_folder, 'args.json')))
            abs_path = os.path.abspath(checkpoint_folder + '/checkpoints/')
            checkpoint = checkpoints.restore_checkpoint(abs_path, target=None)
            if checkpoint is None:
                continue

            params = checkpoint["vec_q"]["params"]
            algo = args["algorithm"]
            weights = params["params"]["VmapSoftQNetwork_0"]["Dense_0"]["kernel"]
            action_weights = weights[:, :, :]
            difficulty = folder.split("-")[-2]

            weights_dict[algo + "_" +  difficulty] = action_weights.flatten()
            
            
    return weights_dict

    
def visualize_weights(weights_dict, ax, labels=None):
    color_map = { "EDAC" : "blue", "CQL" : "red", "EXPERT" : "blue",
                 "REPLAY" : "red" }

    for i, (label, weights) in enumerate(sorted(weights_dict.items())):
        kde = gaussian_kde(weights)
        x_vals = np.linspace(-0.5, 0.5, 200)
        y_vals = kde(x_vals)

        label = labels[i] if labels is not None else label
        ax.plot(x_vals, y_vals, label=label, color=color_map[label])



if __name__ == "__main__":
    dir1 = "vis_data/3d_vis/expertvis-qplot/edac"
    dir2 = "vis_data/3d_vis/expertvis-qplot/compare"

    all_folders = collect_folders([dir1, dir2])
    cmap = get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=set_size(width_fraction=0.8,
                                                    height_fraction=0.15),
                                                    sharey=True)
    weights1 = load_checkpoints(dir1)
    visualize_weights(weights1, axes[0], labels=["EXPERT", "REPLAY"])
    weights2 = load_checkpoints(dir2)
    visualize_weights(weights2, axes[1], labels=["EDAC", "CQL"])
    axes[0].set_ylabel("Density")
    fig.subplots_adjust(bottom=0.15)  
    fig.text(0.5, -0.10, "Weight Value", ha="center")
    plt.savefig("figures/weight_distributions.pdf", dpi=300, bbox_inches="tight")
