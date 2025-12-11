import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import uniform


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale
    return _init


def he_normal():
    return nn.initializers.variance_scaling(
        scale=2.0,
        mode="fan_in",
        distribution="truncated_normal" 
    )
