from functools import partial

import jax
import jax.numpy as jnp
import optax

# Assortion of JIT-compilable pretraining functions for the model, consider 
# different regularizations for critic, actor, etc.
# should we use MC returns as well?

# distributional pretraining?

# should be vmappable over batches like actor/critic losses, like so
# make a function that selects the right pretraining loss based on argument,
# sort of a "loss factory".


# Add lagrangian for scheduling of losses

def pretrain_loss_factory(args, actor_apply_fn, q_apply_fn, alpha_apply_fn):

    """
        A factory for loss functions, accepts all fixed hyperparameters and
        model forwards as arguments, closures over them and returns pretraining
        loss based on args.pretrain_loss

        Every loss should follow the signature:
            loss_name(agent_state, rng, batch):
               and return a loss that can be differentiated + JIT compiled
    """

    """
        Agent losses 
    """
    def soft_bc_loss(agent_state, rng, batch):
        @partial(jax.value_and_grad, argnums=0)
        def _vmapped_loss(actor_params, rng, batch):
            alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))
            def _loss_fn(rng, transition):
                pi = actor_apply_fn(actor_params, transition.obs)
                _, log_pi = pi.sample_and_log_prob(seed=rng)
                log_action = pi.log_prob(transition.action)

                # maximize logprob of dataset action + entropy
                return -log_action.sum(-1) + alpha * log_pi.sum(-1)

            rng = jax.random.split(rng, args.batch_size)
            loss = jax.vmap(_loss_fn)(rng, batch)
            return loss.mean()

        return _vmapped_loss(agent_state.actor.params, rng, batch)

    """ 
        Critic losses
    """

    def sarsa_loss(agent_state, rng, batch):
        # calculate targets w. no grad
        bootstrap_q = q_apply_fn(agent_state.vec_q_target.params, batch.next_obs, batch.next_action)
        target = jnp.expand_dims(batch.reward, -1) + \
                 args.gamma * (1 - jnp.expand_dims(batch.done, -1)) * bootstrap_q
    
        @partial(jax.value_and_grad, argnums=0)
        def _loss_fn(critic_params, rng, batch):
            q_values = q_apply_fn(critic_params, batch.obs, batch.action)
            sarsa_loss = jnp.square(q_values - target)
            return sarsa_loss.mean()

        return _loss_fn(agent_state.vec_q.params, rng, batch)


    """
        Registering losses, lookup
    """

    loss_dict = {
            "bc+sarsa": (soft_bc_loss, sarsa_loss),
    }

    if args.pretrain_loss in loss_dict:
        return loss_dict[args.pretrain_loss]

    else:
        raise ValueError(f"Unknown pretraining loss: {args.pretrain_loss}. "
                         f"Available losses: {list(loss_dict.keys())}.")


def make_pretrain_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset):


    _actor_loss_fn, _critic_loss_fn = pretrain_loss_factory(
        args, actor_apply_fn, q_apply_fn, alpha_apply_fn
    )

    """
        Pretraining step function, accepts agent state and dataset,
        returns loss and updated agent state.
    """

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )

        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        # --- Update alpha ---
        @jax.value_and_grad
        def _alpha_loss_fn(params, rng):
            def _compute_entropy(rng, transition):
                pi = actor_apply_fn(agent_state.actor.params, transition.obs)
                _, log_pi = pi.sample_and_log_prob(seed=rng)
                return -log_pi.sum()

            log_alpha = alpha_apply_fn(params)
            rng = jax.random.split(rng, args.batch_size)
            entropy = jax.vmap(_compute_entropy)(rng, batch).mean()
            target_entropy = -batch.action.shape[-1]
            return log_alpha * (entropy - target_entropy)

        rng, rng_alpha = jax.random.split(rng)
        alpha_loss, alpha_grad = _alpha_loss_fn(agent_state.alpha.params, rng_alpha)
        updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
        agent_state = agent_state._replace(alpha=updated_alpha)

        # --- Update actor ---
        rng, rng_actor = jax.random.split(rng)
        actor_loss, actor_grad = _actor_loss_fn(agent_state, rng_actor, batch)
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        # --- Update critic ---
        rng, rng_critic = jax.random.split(rng)
        critic_loss, critic_grad = _critic_loss_fn(agent_state, rng_critic, batch)
        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
        agent_state = agent_state._replace(vec_q=updated_q)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.vec_q.params,
            agent_state.vec_q_target.params,
            args.polyak_step_size,
        )

        loss = {
                "critic_loss" : critic_loss,
                "actor_loss" : actor_loss,
        }

        return (rng, agent_state), loss


    return _train_step



