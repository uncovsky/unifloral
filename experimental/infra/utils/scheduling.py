# Some schedules for lagrangian parameters
import jax.numpy as jnp

def linear_schedule(start, end, max_steps, offset=0):
    """Linear schedule from start to end with max steps."""
    def schedule(step):
            step -= offset
            return start + (end - start) * (step / max_steps)
    return schedule

    
def constant_schedule(value):
    """Constant schedule."""
    def schedule(step):
        return value
    return schedule


def exponential_schedule(start, min_value, decay_rate, decay_steps, offset=0):
    def schedule(step):
        step = step - offset
        exponent = jnp.floor_divide(step, decay_steps)
        decayed = start / (decay_rate ** exponent)
        return jnp.maximum(min_value, decayed)
    return schedule


def combined_schedule(schedules, steps):
    # traceable combined schedule with time intervals, steps starts with 0, ends implicitly with inf
    # [ConstantSchedule(2.0), ConstantSchedule(1.0)], [0, 10000]
    # First 10k steps with 2.0, then 1.0 afterwards

    def schedule(step):
        vals = jnp.array([s(step) for s in schedules])
        idx = jnp.sum(step >= jnp.array(steps)) - 1
        idx = jnp.clip(idx, 0, len(schedules) - 1)
        return vals[idx]

    return schedule
