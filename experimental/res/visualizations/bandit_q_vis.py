import distrax
import numpy as np
import jax
import gymnasium as gym
import jax.numpy as jnp
import json

from flax.training import checkpoints
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from functools import partial
import seaborn as sns
import os
import matplotlib.pyplot as plt
import yaml

def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale
    return _init

class TanhGaussianActor(nn.Module):
    num_actions: int
    log_std_max: float = 2.0
    log_std_min: float = -5.0

    @nn.compact
    def __call__(self, x):
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        log_std = nn.Dense(
                self.num_actions, kernel_init=sym(1e-3), bias_init=sym(1e-3)
                )(x)
        std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
        mean = nn.Dense(
                self.num_actions, kernel_init=sym(1e-3), bias_init=sym(1e-3)
                )(x)
        pi = distrax.Transformed(
                distrax.Normal(mean, std),
                distrax.Tanh(),
                )
        return pi


class SoftQNetwork(nn.Module):
    depth: int = 3
    critic_norm: str = "none"
    learnable: bool = True
    @nn.compact
    def __call__(self, obs, action, train=True):
        x = jnp.concatenate([obs, action], axis=-1)

        for _ in range(self.depth):
            layer = nn.Dense(256, bias_init=constant(0.1))

            if self.learnable:
                if self.critic_norm == "layer":
                    x = layer(x)
                    x = nn.LayerNorm()(x)
                else:
                    # no normalization
                    x = layer(x)

            else:
                # just forward
                x = layer(x)

            x = nn.relu(x)

        # For learnable Q-nets, we use a different last layer init
        if self.learnable:
            last_layer = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))
        else:
            last_layer = nn.Dense(1, kernel_init=he_normal, bias_init=sym(3e-3))
        q = last_layer(x)

        return q.squeeze(-1)


class VectorQ(nn.Module):
    num_critics: int
    critic_norm: str = "none"
    @nn.compact
    def __call__(self, obs, action, train=True):
        vmap_critic = nn.vmap(
                partial(SoftQNetwork, critic_norm=self.critic_norm, learnable=True), # all learnable
                variable_axes={"params": 0, "batch_stats" : 0},  # Parameters not shared between critics
                split_rngs={"params": True, "dropout": True},  # Different initializations
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action, train)
        return q_values





def get_q_vals_vis(ckpt_dir='./bandit_checkpoints'):

    for folder in os.listdir(ckpt_dir):
        print(f'Loading folder {folder}')
        for checkpoint in os.listdir(os.path.join(ckpt_dir, folder)):
            checkpoint_folder = os.path.join(ckpt_dir, folder, checkpoint)

            args = json.load(open(os.path.join(checkpoint_folder, 'args.json'), 'r'))
            abs_path = os.path.abspath(checkpoint_folder + '/checkpoints/')
            checkpoint = checkpoints.restore_checkpoint(abs_path, target=None)

            algo = args['algorithm']
            num_critics = args['num_critics']
            critic_norm = args['critic_norm']
            ens_lab = args.get('reg_lagrangian', 0.0)
            critic_lag = args.get('critic_lagrangian', 0.0)
            critic_reg = args.get('critic_regularizer', 'none')

            state_dim = 1
            action_dim = 1

        
            # --- Setup Seaborn and fonts for LaTeX-style plots ---
            sns.set(style="whitegrid")
            plt.rcParams.update({
                "font.size": 14,        # large font for LaTeX
                "axes.labelsize": 16,
                "axes.titlesize": 16,
                "legend.fontsize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12
            })

            # --- Data preparation ---
            actions = np.linspace(-1, 1, 1000).reshape(-1, 1)
            expert_q_values = np.array([1 if abs(a) >= 0.5 else -1 for a in actions], dtype=np.float32)

            state_tensor = jnp.ones((1000, state_dim), dtype=jnp.float32)
            actions_tensor = jnp.array(actions, dtype=jnp.float32)

            # --- Model inference ---
            model = VectorQ(num_critics=num_critics, critic_norm=critic_norm)

            if 'batch_stats' in checkpoint['vec_q']:
                vec_q_params = checkpoint['vec_q']
                q_values = model.apply({'params': vec_q_params['params'], 'batch_stats': {}}, 
                                       state_tensor, actions_tensor, train=False, mutable=False)
            else:
                vec_q_params = checkpoint['vec_q']['params']
                q_values = model.apply(vec_q_params, state_tensor, actions_tensor)

            # --- Bias calculation ---
            positive_expert = expert_q_values >= 0
            negative_expert = expert_q_values < 0

            bias_pos = jnp.mean(q_values[positive_expert], axis=1) - expert_q_values[positive_expert]
            bias_neg = jnp.mean(q_values[negative_expert], axis=1) - expert_q_values[negative_expert]

            print(f'For {checkpoint_folder} with norm {critic_norm}, bias pos: {bias_pos.mean()}, bias neg: {bias_neg.mean()}')

            # --- Plotting using Seaborn ---
            fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
            title = f"{algo}_critics_{num_critics}_lagc_{critic_lag}_lage_{ens_lab}_norm_{critic_norm}_reg_{critic_reg}"

            # Expert line
            sns.lineplot(x=actions.flatten(), y=expert_q_values, label='Expert', color='black', linestyle='dashed', linewidth=2)

            # Critics lines (dotted)
            if algo == "cql":
                num_critics = 2
            for i in range(num_critics):
                sns.lineplot(x=actions.flatten(), y=q_values[:, i], label=f'Critic {i}', linestyle='--', linewidth=1.5)

            ax1.set_xlabel('Action')
            ax1.set_ylabel('Q Value')
            ax1.legend()
            ax1.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)  # lighter and faint grid
            plt.tight_layout()
            plt.savefig(f'{title}_q_vals.png', dpi=300)
            plt.clf()

            # ----------------------------------------------------------------
            # --- SECOND FIGURE: Q-values + actor action histogram ---
            # ----------------------------------------------------------------

            # Load actor and sample actions
            actor = TanhGaussianActor(num_actions=action_dim)
            actor_params = checkpoint['actor']['params']

            pi = actor.apply(actor_params, state_tensor)
            sampled_actions = pi.sample(seed=jax.random.PRNGKey(0))
            action_list = np.array(sampled_actions).flatten()

            # Create new figure
            fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=400)

            # Expert line
            sns.lineplot(
                x=actions.flatten(),
                y=expert_q_values,
                label='Expert',
                color='black',
                linestyle='--',
                linewidth=2,
                ax=ax2
            )

            # Critics lines
            for i in range(num_critics):
                sns.lineplot(
                    x=actions.flatten(),
                    y=q_values[:, i],
                    label=f'Critic {i}',
                    linestyle=':',
                    linewidth=2.5,
                    ax=ax2
                )

            # Twin axis for histogram
            ax_hist = ax2.twinx()
            sns.histplot(
                action_list,
                bins=30,
                color='tab:green',
                alpha=0.25,
                kde=False,
                stat='density',
                ax=ax_hist,
                label='Actor Action Distribution'
            )
            scale = 0.5
            for patch in ax_hist.patches:
                patch.set_height(patch.get_height() * scale)
            ax_hist.set_ylabel('')                      
            ax_hist.set_yticks([])                       
            ax_hist.grid(False)                     

            ax2.set_xlabel('Action')
            ax2.set_ylim(-5, 2)
            ax2.set_ylabel('Q Value')
            #ax_hist.set_ylabel('Sample Density')
            # Combine legends from both axes
            lines_labels = [ax.get_legend_handles_labels() for ax in [ax2, ax_hist]]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            ax2.legend(lines, labels, loc='upper left')

            ax2.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{title}_q_vals_with_hist.png', dpi=400)
            plt.clf()


    

#get_q_vals_vis("./bandit_checkpoints/new")
get_q_vals_vis("../../checkpoints/bandit")
