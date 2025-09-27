import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from jax.nn.initializers import constant
from infra.models.initialization import sym

# Taken from Unifloral

# Standard truncated gaussian policy with tanh squashing
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


# Module for learning the entropy coefficient
class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
                "log_ent_coef",
                init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
                )
        return log_ent_coef
