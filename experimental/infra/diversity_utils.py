import jax
import jax.numpy as jnp
from infra.offline_dataset_wrapper import OfflineDatasetWrapper

"""
    A collection of utility functions used to measure
    ensemble diversity in experiments.
"""

def diversity_loss(q_apply_fn, agent_state, obs, actions, num_critics):
    """
        Compute EDAC diversity loss for a batch of s,a pairs
    """
    def _diversity_loss_fn(obs, action):

        """
            Compute <nabla_a Q_i(s,a), nabla_a Q_j(s,a)> for all i,j
            for a single (s,a) pair.
        """
        # shape (E, A) ensemble outputs, A inputs (action)
        action_jac = jax.jacrev(q_apply_fn, argnums=2)(agent_state.vec_q.params, obs, action)
        # shape (E,A), normalized gradients for each ensemble member
        action_jac /= jnp.linalg.norm(action_jac, axis=-1, keepdims=True) + 1e-6
        # shape (E,E) pairwise diversity loss
        div_loss = action_jac @ action_jac.T
        # Mask diagonal 
        div_loss *= 1.0 - jnp.eye(num_critics)
        return div_loss.sum()

    # vmap over whole batch
    diversity_loss = jax.vmap(_diversity_loss_fn)(obs, actions)
    diversity_loss = diversity_loss.mean() / (num_critics - 1)
    return diversity_loss


def prepare_ood_dataset(rng, dataset_name="walker2d-expert-v2", ood_samples=50):
    """
        Samples a set of ood_samples (s,a) pairs from the given dataset.

        The intended usage is to train on a dataset (e.g. walker2d-medium-v2)
            and use these ood states and actions to evaluate ensemble
            diversity on unseen states.
    """

    walker_expert_wrapper = OfflineDatasetWrapper(source="d4rl",
                                                  dataset=dataset_name)
    data = walker_expert_wrapper.get_dataset()
    size = len(data["observations"])

    indices = jax.random.randint(rng, (ood_samples,), 0, size)
    ood_obs = jnp.array(data["observations"])[indices]
    ood_actions = jnp.array(data["actions"])[indices]

    return ood_obs, ood_actions


def compute_qvalue_statistics(q_apply_fn, agent_state, obs, actions):
    """
        Computes statistics for the q-ensemble, given a batch of s,a pairs
    """

    # These have shape [B,E]
    q_values = q_apply_fn(agent_state.vec_q.params, obs, actions)

    q_std = jnp.std(q_values, axis=1).mean()
    q_min = jnp.min(q_values, axis=1).mean()
    q_mean = jnp.mean(q_values)

    return {"std": q_std,
            "min": q_min,
            "mean": q_mean}




def get_diversity_statistics(q_apply_fn, actor_apply_fn, agent_state, rng, obs, actions):

    """
        Computes quantities that are logged to measure ensemble diversity for a
        given batch of s,a pairs.

        Assumes all actions are in [-1,1] range.
    """
    
    batch_stats = compute_qvalue_statistics(q_apply_fn, agent_state, obs, actions)

    # sample actions from actor
    rng, rng_pi = jax.random.split(rng)
    pi_actions = actor_apply_fn(agent_state.actor.params, obs).sample(seed=rng_pi)

    pi_stats = compute_qvalue_statistics(q_apply_fn, agent_state, obs, pi_actions)

    # add gaussian noise to sampled actions, std=0.1 and std=0.5
    rng_unif, rng_noise, rng_noise2 = jax.random.split(rng, 3)
    noise_small = jax.random.normal(rng_noise, pi_actions.shape) * 0.1
    noise_large = jax.random.normal(rng_noise2, pi_actions.shape) * 0.5
    perturbed_actions_small = jnp.clip(pi_actions + noise_small, -1.0, 1.0)
    perturbed_actions_large = jnp.clip(pi_actions + noise_large, -1.0, 1.0)

    noise_small_stats = compute_qvalue_statistics(q_apply_fn, agent_state, obs, perturbed_actions_small)
    noise_large_stats = compute_qvalue_statistics(q_apply_fn, agent_state, obs, perturbed_actions_large)

    # uniform actions
    uniform_actions = jax.random.uniform(rng_unif, pi_actions.shape, minval=-1.0, maxval=1.0)
    uniform_stats = compute_qvalue_statistics(q_apply_fn, agent_state, obs, uniform_actions)

    # Collect ensemble devications for each of the action types
    diversity_stats = {
        "batch_q_std": batch_stats["std"],
        "pi_q_std": pi_stats["std"],
        "noise_small_q_std": noise_small_stats["std"],
        "noise_large_q_std": noise_large_stats["std"],
        "uniform_q_std": uniform_stats["std"],
        "batch_q_min": batch_stats["min"],
        "pi_q_min": pi_stats["min"],
        "noise_small_q_min": noise_small_stats["min"],
        "noise_large_q_min": noise_large_stats["min"],
        "uniform_q_min": uniform_stats["min"],
        "batch_q_mean": batch_stats["mean"],
        "pi_q_mean": pi_stats["mean"],
        "noise_small_q_mean": noise_small_stats["mean"],
        "noise_large_q_mean": noise_large_stats["mean"],
        "uniform_q_mean": uniform_stats["mean"],
    }

    return diversity_stats









