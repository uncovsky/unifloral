from functools import partial

import jax
import jax.numpy as jnp
import optax


# JIT-compilable regularizer functions that can be used with any 
# offline A-C algorithm.
def regularizer_factory(args, actor_apply_fn, q_apply_fn):

    """
        A factory for regularization functions, accepts all fixed hyperparameters and
        model forwards as arguments, closures over them and returns regularization based on 
        args.critic_regularizer.

        The loss uses args.critic_lagrangian based on the type of loss.

        The factory closures over global parameters (args, forwards)

        Every loss follows the signature:
            loss_name(agent_state, rng, batch):
               and returns a loss function that can be differentiated + traced,
               used in the critic update step.
    """

    """
        Regularizers
    """

    def noop_loss(agent_state, rng, batch):
        """
        No-op loss, used when args.critic_regularizer is 'none'
        """
        def _noop_loss_fn(critic_params, rng, batch):
            return jnp.array(0.0)

        return _noop_loss_fn
    
    def pbrl_regularizer(agent_state, rng, batch):

        # nondifferentiated part
        rng, rng_curr, rng_next = jax.random.split(rng, 3)

        pi_curr = actor_apply_fn(agent_state.actor.params, batch.obs)
        pi_next = actor_apply_fn(agent_state.actor.params, batch.next_obs)

        ood_actions, _ = pi_curr.sample_and_log_prob(seed=rng_curr,
                                                     sample_shape=(args.critic_regularizer_parameter,))
        ood_actions_next, _ = pi_next.sample_and_log_prob(seed=rng_next,
                                                          sample_shape=(args.critic_regularizer_parameter,))

        # [action_num, B, action_dim] -> [B, action_num, action_dim]
        ood_actions = jnp.swapaxes(ood_actions, 0, 1)
        ood_actions_next = jnp.swapaxes(ood_actions_next, 0, 1)

        # Make a (B, num_samples, obs_dim) 
        states = jnp.expand_dims(batch.obs, axis=1).repeat(args.critic_regularizer_parameter, axis=1)
        next_states = jnp.expand_dims(batch.next_obs,axis=1).repeat(args.critic_regularizer_parameter, axis=1)
        
        def _loss_fn(q_pred, critic_params, rng, batch):

            pi_curr = actor_apply_fn(agent_state.actor.params, batch.obs)

            # Get Q vals for ood actions
            q_ood = q_apply_fn(critic_params, 
                               states, ood_actions)

            q_ood_next = q_apply_fn(critic_params,
                                    next_states, ood_actions_next)

            std_q_ood = jnp.std(q_ood, axis=1, keepdims=True)
            std_q_ood_next = jnp.std(q_ood_next, axis=1, keepdims=True)

            # Q - beta_ood * std + clip
            ood_q_target = q_ood - args.critic_lagrangian * std_q_ood
            ood_q_target = jnp.maximum(ood_q_target, 0.0)
            ood_q_target = jax.lax.stop_gradient(ood_q_target)

            # PBRL fixes this at 0.1 for some reason, but we use the lagrangian
            ood_q_target_next = q_ood_next - args.critic_lagrangian * std_q_ood_next
            ood_q_target_next = jnp.maximum(ood_q_target_next, 0.0)
            ood_q_target_next = jax.lax.stop_gradient(ood_q_target_next)

            # Sum over ensemble, mean over batch and samples
            ood_loss = jnp.square(q_ood - ood_q_target).sum(axis=1).mean()
            ood_loss += jnp.square(q_ood_next - ood_q_target_next).sum(axis=1).mean()

            return ood_loss

        return _loss_fn


    def msg_regularizer(agent_state, rng, batch):

        # nondifferentiated part
        rng, rng_curr = jax.random.split(rng, 2)
        pi_curr = actor_apply_fn(agent_state.actor.params, batch.obs)

        ood_actions, _ = pi_curr.sample_and_log_prob(seed=rng_curr,
                                                     sample_shape=(args.critic_regularizer_parameter,))

        # [action_num, B, action_dim] -> [B, action_num, action_dim]
        ood_actions = jnp.swapaxes(ood_actions, 0, 1)
        # Make a (B, num_samples, obs_dim) 
        states = jnp.expand_dims(batch.obs, axis=1).repeat(args.critic_regularizer_parameter, axis=1)

        def _loss_fn(q_pred, critic_params, rng, batch):
            # Get Q vals for ood actions
            q_ood = q_apply_fn(critic_params, 
                               states, ood_actions)
            # [B, num_samples, E]
            loss = q_ood.mean() - q_pred.mean()
            # Sum over E, mean over B and samples
            # Apply lagrangian here
            loss = args.critic_lagrangian * loss

            return loss

        return _loss_fn


    def cql_regularizer(agent_state, rng, batch):
        rng, rng_pi, rng_next = jax.random.split(rng, 3)

        pi = actor_apply_fn(agent_state.actor.params, batch.obs)
        pi_next = actor_apply_fn(agent_state.actor.params, batch.next_obs)

        pi_actions, _ = pi.sample_and_log_prob(seed=rng_pi)
        pi_next_actions, _ = pi_next.sample_and_log_prob(seed=rng_next)

        rng, rng_random = jax.random.split(rng)

        cql_random_actions = jax.random.uniform(
            rng_random, shape=batch.action.shape, minval=-args.action_scale,
            maxval=args.action_scale
        )


        def _loss_fn(q_pred, critic_params, rng, batch):
            
            q_pi = q_apply_fn(critic_params,
                              batch.obs, pi_actions)

            q_pi_next = q_apply_fn(critic_params,
                                   batch.next_obs, pi_next_actions)

            q_random = q_apply_fn(critic_params,
                                  batch.obs, cql_random_actions)

            all_qs = jnp.stack([q_pred, q_pi, q_pi_next, q_random], axis=1)
            q_ood = jax.scipy.special.logsumexp(all_qs / args.critic_regularizer_parameter, axis=1).sum(-1)
            q_ood = q_ood * args.critic_regularizer_parameter

            min_q_loss = (q_ood.mean() - q_pred.mean()) * args.critic_lagrangian

            return min_q_loss

        return _loss_fn


    loss_dict = {
            "none": noop_loss,
            "pbrl": pbrl_regularizer,
            "msg": msg_regularizer,
            "cql": cql_regularizer
    }

    if args.critic_regularizer in loss_dict:
        return loss_dict[args.critic_regularizer]

    else:
        raise ValueError(f"Unknown ensemble regularizer: {args.ensemble_regularizer}. "
                         f"Available regularizers: {list(loss_dict.keys())}.")



def select_ood_regularizer(args, actor_apply_fn, q_apply_fn):
    """
        Helper function to select regularizer
    """
    return regularizer_factory(args, actor_apply_fn, q_apply_fn)
