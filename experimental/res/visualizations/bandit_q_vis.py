import numpy as np
import jax
import gymnasium as gym
import jax.numpy as jnp
import json

from flax.training import checkpoints
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from functools import partial

import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml
import seaborn as sns

def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale
    return _init



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





def get_q_vals_vis(ckpt_dir='./bandit_checkpoints', file=None):

    for algorithm_folder in os.listdir(ckpt_dir):

        for run_folder in os.listdir(os.path.join(ckpt_dir, algorithm_folder)):

            run_folder_dir = os.path.join(ckpt_dir, algorithm_folder, run_folder)
            checkpoint_dir = os.path.join(run_folder_dir, 'checkpoints')

            args = json.load(open(os.path.join(run_folder_dir, 'args.json'), 'r'))

            abs_path = os.path.abspath(checkpoint_dir)
            print("parsing checkpoint dir:", checkpoint_dir)
            checkpoint = checkpoints.restore_checkpoint(abs_path, target=None)

            algo = args['algorithm']
            num_critics = args['num_critics']
            critic_norm = args['critic_norm']
            ens_lab = args.get('reg_lagrangian', 0.0)
            critic_lag = args.get('critic_lagrangian', 0.0)

            state_dim = 1
            action_dim = 1

        
            actions = np.linspace(-1, 1, 1000).reshape(-1, 1)
            q_values = []
            expert_q_values = [ 1 if abs(a) >= 0.5 else -1 for a in actions]

            expert_q_values = jnp.array(expert_q_values, dtype=jnp.float32)


            state_tensor = jnp.ones((1000, state_dim), dtype=jnp.float32)
            actions_tensor = jnp.array(actions, dtype=jnp.float32)

            model = VectorQ(num_critics=num_critics, critic_norm=critic_norm)
            if 'batch_stats' in checkpoint['vec_q']:
                vec_q_params = checkpoint['vec_q']
                q_values = model.apply({'params': vec_q_params['params'],
                                        'batch_stats': {}}, 
                                       state_tensor, actions_tensor, train=False, mutable=False)
            else:
                vec_q_params = checkpoint['vec_q']['params']
                q_values = model.apply(vec_q_params, state_tensor, actions_tensor)
                q_value_mins = jnp.min(q_values, axis=1)

            data = {
                'actions': actions.flatten(),
                'expert': expert_q_values
            }

            for i in range(num_critics):
                data[f'critic_{i}'] = q_values[:, i]

            # get bias of q-values vs expert q-values
            positive_expert = expert_q_values >= 0
            negative_expert = expert_q_values < 0

            bias_pos = jnp.min(q_values[positive_expert], axis=1) - expert_q_values[positive_expert]
            bias_neg = jnp.min(q_values[negative_expert], axis=1) - expert_q_values[negative_expert]


            def make_title(algo_folder):
                name = algo_folder.split('_')[0]

                title = f"{name}, num_critics={num_critics}"

                if name == 'edac':
                    title += f", ens_lab={ens_lab}"
                if name in ['msg', 'cql', 'pbrl']:
                    title += f", critic_lag={critic_lag}"
                if name in ['msg', 'pbrl']:
                    title += f", critic_norm={critic_norm}"

                return title

            title = make_title(algorithm_folder)
            f.write("{title}, {bias_pos}, {bias_neg}\n".format(title=title, bias_pos=jnp.mean(bias_pos), bias_neg=jnp.mean(bias_neg)))


            df_wide = pd.DataFrame(data)

            df = df_wide.melt(id_vars=['actions'], var_name='source', value_name='Q value')

            # Split expert vs critics
            df_expert = df[df["source"] == "expert"]
            df_critics = df[df["source"] != "expert"]

            df_min_critics = df_critics.groupby("actions").min().reset_index()




            plt.figure(figsize=(10, 6))

            # Plot expert manually → solid black
            plt.plot(df_expert["actions"], df_expert["Q value"], color="black", linestyle="solid", label="expert")

            sns.lineplot(
                data=df_critics,
                x="actions", y="Q value",
                hue="source",
                style="source",
                dashes=[(1, 2)],   # make all dotted
                palette="tab10"
            )


            plt.xlabel("Action")
            plt.ylabel("Q-values")

            plt.title("Critic values for " + title)

            os.makedirs("plots", exist_ok=True)
            plt.savefig(os.path.join("plots", f"{title}_all_critics.png"))


            # plot min of critics
            plt.figure(figsize=(10, 6))
            plt.plot(df_expert["actions"], df_expert["Q value"], color="black", linestyle="solid", label="expert")

            sns.lineplot(
                data=df_min_critics,
                x="actions",
                y="Q value",
                color="red",
                linestyle="dashed",
                linewidth=2,
                label="min critics"
            )

            plt.xlabel("Action")
            plt.title("Min Q-values for " + make_title(algorithm_folder))
            plt.ylabel("Q-values")
            plt.savefig(os.path.join("plots", f"{title}_min_critics.png"))



if __name__ == "__main__":
    os.makedirs("bandit_vis/", exist_ok=True)
    with open(os.path.join("bandit_vis", f"bias.csv"), 'w+') as f:
        f.write("title, bias_pos, bias_neg\n")
        #get_q_vals_vis("./bandit_checkpoints/new")
        get_q_vals_vis("../checkpoints/bandit", file=f)
