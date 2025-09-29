import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training.train_state import TrainState

"""
    UNUSED (stateful normalization layers are not used in the end)
"""

def create_aug_train_state(args, rng, network, dummy_input, lr=None):
    variables = network.init(rng, *dummy_input)
    return AugmentedTrainState.create(
            apply_fn=network.apply,
            params=variables.get("params"),
            batch_stats = variables.get("batch_stats", {}),
            tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
            )

class AugmentedTrainState(TrainState):
    batch_stats: any = None

def NormalizationWrapper(q_net):
    """
        Wraps around a forward of a Q-network, enabling us to handle
        batch_stats properly no matter if normalization layers are used or not.
    """
    def q_apply_fn(params, batch_stats, obs, action, train=True):
        """
            Args:
                params: parameters of the Q-network
                batch_stats: batch statistics (for normalization layers)
                obs: observations
                action: actions
                train: whether in training mode or not
        """

        if train:
            # Return new batch_stats
            outputs, new_state = q_net.apply(
                {"params": params, "batch_stats": batch_stats},
                obs,
                action,
                train=True,
                mutable=["batch_stats"]
            )

            return outputs, new_state["batch_stats"]
        else:
            # Return old batch_stats unchanged
            outputs = q_net.apply(
                {"params": params, "batch_stats": batch_stats},
                obs,
                action,
                train=False,
                mutable=False
            )
            return outputs, batch_stats

    return q_apply_fn


