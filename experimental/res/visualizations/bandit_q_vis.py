import numpy as np
import jax
import gymnasium as gym
import jax.numpy as jnp
import json

from flax.training import checkpoints
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from functools import partial

import os
import matplotlib.pyplot as plt
import yaml
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





def get_q_vals_vis(ckpt_dir='./bandit_checkpoints'):

    for folder in os.listdir(ckpt_dir):
        for checkpoint in os.listdir(os.path.join(ckpt_dir, folder)):
            checkpoint_folder = os.path.join(ckpt_dir, folder, checkpoint)

            args = json.load(open(os.path.join(checkpoint_folder, 'args.json'), 'r'))
            abs_path = os.path.abspath(checkpoint_folder)
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




            # get bias of q-values vs expert q-values
            positive_expert = expert_q_values >= 0
            negative_expert = expert_q_values < 0

            bias_pos = jnp.mean(q_values[positive_expert], axis=1) - expert_q_values[positive_expert]
            bias_neg = jnp.mean(q_values[negative_expert], axis=1) - expert_q_values[negative_expert]

            print(f'For {checkpoint_folder} with norm {critic_norm}, bias pos: {bias_pos.mean()}, bias neg: {bias_neg.mean()}')

            plt.plot(actions, expert_q_values, label='expert', color='black',
                     linestyle='dashed')
            for i in range(num_critics):
                plt.plot(actions, q_values[:, i], label=f'critic {i}')
            plt.xlabel('action')
            title = "{algo}_critics_{num_critics}_lagc_{critic_lag}_lage_{ens_lab}_norm_{critic_norm}".format(algo=algo, 
                                                                                                              num_critics=num_critics, 
                                                                                                              critic_lag=critic_lag, 
                                                                                                              ens_lab=ens_lab, 
                                                                                                              critic_norm=critic_norm)
            plt.title(f'Ensemble Q values {algo}, lag_c = {critic_lag}, lag_e = {ens_lab}, norm = {critic_norm}')
            # yaxis limit for better visualization
            plt.ylabel('Q value')
            plt.legend()
            # plt.savefig(f'{title}_q_vals.png')
            plt.show()
            plt.clf()



    

#get_q_vals_vis("./bandit_checkpoints/new")
get_q_vals_vis("./bandit_checkpoints/bandit_pbrl")
