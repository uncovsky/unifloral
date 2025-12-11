import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from jax.nn.initializers import constant
from infra.models.initialization import sym

# Adapted from Unifloral

"""
    Actor nets Support ONLY [-1, 1]^d action spaces so far.
"""
class TanhGaussianActor(nn.Module):
    num_actions: int
    per_state_std: bool = True
    log_std_max: float = 2.0
    log_std_min: float = -5.0

    @nn.compact
    def __call__(self, x, eval=False):
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)

        mean = nn.Dense(
                self.num_actions, kernel_init=sym(1e-3), bias_init=sym(1e-3)
                )(x)

        if eval:
            # Deterministic policy for eval
            pi = distrax.Deterministic(jnp.tanh(mean))
            return pi

        if self.per_state_std:
            log_std = nn.Dense(
                    self.num_actions, kernel_init=sym(1e-3), bias_init=sym(1e-3)
                    )(x)

            std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
            pi = distrax.Transformed(
                    distrax.Normal(mean, std),
                    distrax.Tanh(),
            )

        else:

            """
                AWAC uses a clipped gaussian, where mean is first passed
                through tanh. STD is state-independent, learned parameter.
            """
            log_std = self.param(
                "log_std",
                init_fn=lambda key: jnp.zeros(self.num_actions, dtype=jnp.float32),
            )


            std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
            # Clipped normal distribution. 
            # Action clipping is handled during eval.
            pi = distrax.Normal(jnp.tanh(mean), std)

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
