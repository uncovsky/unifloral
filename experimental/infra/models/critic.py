
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import constant

from infra.models.initialization import sym, he_normal


"""
    Main network class vectorized over.

    Supports normalization (from critic_norm)

    Uses different scaling for last layer on learnable vs prior fns.

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
                    # no normalization
                    x = layer(x)
            else:
                # Non-learnable nets (prior) have no normalization
                x = layer(x)

            x = nn.relu(x)

        # For learnable Q-nets, we use a different last layer init
        if self.learnable:
            last_layer = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))
        else:
            last_layer = nn.Dense(1, kernel_init=he_normal(), bias_init=sym(3e-3))
        q = last_layer(x)

        return q.squeeze(-1)

"""
    Wraps around two soft Q networks, one learnable and one fixed (prior).

    The prior network is scaled by a factor and added to the learnable network.
"""
class RandomizedPriorQNetwork(nn.Module):
    depth: int 
    scale: float  
    critic_norm: str = "none"
    @nn.compact
    def __call__(self, obs, action):
        q_learnable = SoftQNetwork(learnable=True, critic_norm=self.critic_norm,
                                   name="learnable_q_network")(obs, action)
        prior_net = SoftQNetwork(learnable=False, critic_norm="none",
                                 depth=self.depth, name="prior_q_network")
        q_prior = prior_net(obs, action)
        # make sure to not prop grad thru prior net
        q_prior = jax.lax.stop_gradient(q_prior)
        # Combine learnable and prior
        return q_learnable + self.scale * q_prior

"""
    Vectorized wrapper over Q-ensemble.
"""
class VectorQ(nn.Module):
    num_critics: int
    depth : int
    critic_norm: str = "none"
    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
                partial(SoftQNetwork, critic_norm=self.critic_norm,
                                      depth=self.depth,
                                      learnable=True), # all learnable
                variable_axes={"params": 0, "batch_stats" : 0},  # Parameters not shared between critics
                split_rngs={"params": True, "dropout": True},  # Different initializations
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action)
        return q_values

"""
    Vectorized wrapper over Q-ensemble with priors.
"""
class PriorVectorQ(nn.Module):
    num_critics: int
    depth: int
    scale: float
    critic_norm: str = "none"
    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
                partial(RandomizedPriorQNetwork, depth=self.depth,
                                                 scale=self.scale,
                                                 critic_norm=self.critic_norm), # all learnable
                variable_axes={"params": 0, "batch_stats" : 0},  # Parameters not shared between critics
                split_rngs={"params": True, "dropout": True},  # Different initializations
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_critics,
                )
        q_values = vmap_critic()(obs, action)
        return q_values


"""
    State value function network.
"""
class StateValueFunction(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        v = nn.Dense(1)(x)
        return v.squeeze(-1)

"""
    Vectorized wrapper over state value function ensemble.
"""
class VectorV(nn.Module):
    num_values: int
    @nn.compact
    def __call__(self, x):
        vmap_value = nn.vmap(
                StateValueFunction,
                variable_axes={"params": 0, "batch_stats" : 0},  # Parameters not shared between values
                split_rngs={"params": True, "dropout": True},  # Different initializations
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_values,
        )
        v_values = vmap_value()(x)
        return v_values
