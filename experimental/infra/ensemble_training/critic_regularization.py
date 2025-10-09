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

            the returned loss function has the signature:
                (q_pred, critic_params, rng, batch) -> (loss, logs)
                we pass q_pred to avoid recomputing it, since it is frequently used
    """

    """
        Regularizers
    """

    def noop_loss(agent_state, rng, batch):
        """
        No-op loss, used when args.critic_regularizer is 'none'
        """
        def _noop_loss_fn(q_pred, critic_params, rng, batch):
            return jnp.array(0.0), {}

        return _noop_loss_fn
    
    def pbrl_regularizer(agent_state, rng, batch, filtered=False,
                                                  use_next_states=False):

        """
            PBRL regularization
            
            _filtered_ is a flag to use adaptive action filtering, based on the
            std of the Q-values. This is not used in the original PBRL paper,

            _use_next_states_ is a flag to use next states for the second part
            of the loss, original PBRL implementation uses next_state
            penalization with a fixed coefficient of 0.1.
        """
        # nondifferentiated part
        rng_curr, rng_next, rng_unif = jax.random.split(rng, 3)

        # Get actions sampled from pi(s) and pi(s')
        pi_curr = actor_apply_fn(agent_state.actor.params, batch.obs)
        pi_next = actor_apply_fn(agent_state.actor.params, batch.next_obs)

        ood_actions, _ = pi_curr.sample_and_log_prob(seed=rng_curr,
                                                     sample_shape=(args.critic_regularizer_parameter,))
        ood_actions_next, _ = pi_next.sample_and_log_prob(seed=rng_next,
                                                          sample_shape=(args.critic_regularizer_parameter,))

        # [action_num, B, action_dim] -> [B, action_num, action_dim]
        ood_actions = jnp.swapaxes(ood_actions, 0, 1)
        # Add uniformly sampled actions
        unif_actions = jax.random.uniform(
            rng_unif, shape=(batch.action.shape[0], args.critic_regularizer_parameter, batch.action.shape[1]),
            minval=-args.action_scale, maxval=args.action_scale
        )
        ood_actions = jnp.concatenate([ood_actions, unif_actions], axis=1)

        # Make a (B, num_samples, obs_dim) state tensor for calculating Q vals
        states = jnp.expand_dims(batch.obs, axis=1).repeat(2 * args.critic_regularizer_parameter, axis=1)

        if use_next_states:
            next_states = jnp.expand_dims(batch.next_obs,axis=1).repeat(args.critic_regularizer_parameter, axis=1)
            ood_actions_next = jnp.swapaxes(ood_actions_next, 0, 1)
        
        def _loss_fn(q_pred, critic_params, rng, batch):

            def _filter_penalties(q_ood, std_q_ood, quantile):

                """
                    Modifies the OOD penalty only to certain states in the batch,
                    namely those with the highest std of Q-values across the
                    sampled actions.
                """
                # [B, 1]
                mean_stds = jnp.mean(std_q_ood, axis=1)
                # [1, 1]
                state_threshold = jnp.quantile(mean_stds, quantile, axis=0, keepdims=True)
                mask = (mean_stds >= state_threshold).astype(jnp.float32)
                # [B, 1, 1] mask for Q-values
                mask = jnp.expand_dims(mask, axis=-1)

                # [B, num_samples, e]
                filtered_q_ood = q_ood * mask
                # [B, num_samples, 1]
                filtered_std_q_ood = std_q_ood * mask

                # rescale mask to maintain loss scale
                mask *= (1.0 / (1.0 - quantile))

                return filtered_q_ood, filtered_std_q_ood
                

            # Get Q vals for ood actions
            q_ood = q_apply_fn(critic_params, 
                               states, ood_actions)
            std_q_ood = jnp.std(q_ood, axis=-1, keepdims=True)

            if filtered:
                q_ood, std_q_ood = _filter_penalties(q_ood, std_q_ood,
                                                    args.filtering_quantile)
                
            # Q - beta_ood * std + clip
            ood_q_target = q_ood - args.critic_lagrangian * std_q_ood
            ood_q_target = jnp.maximum(ood_q_target, 0.0)
            ood_q_target = jax.lax.stop_gradient(ood_q_target)

            # Sum over ensemble, mean over batch and samples
            ood_loss = jnp.square(q_ood - ood_q_target).sum(axis=-1).mean()

            if use_next_states:

                q_ood_next = q_apply_fn(critic_params,
                                        next_states, ood_actions_next)

                std_q_ood_next = jnp.std(q_ood_next, axis=-1, keepdims=True)

                if filtered:
                    q_ood_next, std_q_ood_next = _filter_penalties(q_ood_next,
                                                                   std_q_ood_next,
                                                                   args.filtering_quantile)
                # PBRL always keeps penalty here at 0.1
                ood_q_target_next = q_ood_next - 0.1 * std_q_ood_next
                ood_q_target_next = jnp.maximum(ood_q_target_next, 0.0)
                ood_q_target_next = jax.lax.stop_gradient(ood_q_target_next)

                ood_loss += jnp.square(q_ood_next - ood_q_target_next).sum(axis=-1).mean()
            else:
                q_ood_next = jnp.array(0.0)
                std_q_ood_next = jnp.array(0.0)
                ood_q_target_next = jnp.array(0.0)

            logs = {
                "pbrl_ood_q_mean": q_ood.mean(),
                "pbrl_ood_q_next_mean": q_ood_next.mean(),
                "pbrl_ood_q_std_mean": std_q_ood.mean(),
                "pbrl_ood_q_next_std_mean": std_q_ood_next.mean(),
                "pbrl_ood_q_target_mean": ood_q_target.mean(),
                "pbrl_ood_q_next_target_mean": ood_q_target_next.mean(),
            }

            return ood_loss, logs

        return _loss_fn

    def msg_regularizer(agent_state, rng, batch):

        pi_curr = actor_apply_fn(agent_state.actor.params, batch.obs)

        ood_actions, _ = pi_curr.sample_and_log_prob(seed=rng,
                                                     sample_shape=(args.critic_regularizer_parameter,))

        # [action_num, B, action_dim] -> [B, action_num, action_dim]
        ood_actions = jnp.swapaxes(ood_actions, 0, 1)
        states = jnp.expand_dims(batch.obs, axis=1).repeat(args.critic_regularizer_parameter, axis=1)


        def _loss_fn(q_pred, critic_params, rng, batch):
            # Get Q vals for ood actions
            q_ood = q_apply_fn(critic_params, 
                               states, ood_actions)
            # [B, num_samples, E]
            ood_mean = q_ood.mean()
            pred_mean = q_pred.mean()

            loss = ood_mean - pred_mean

            # Apply lagrangian here
            loss = args.critic_lagrangian * loss

            logs = {
                "msg_ood_q_mean": ood_mean,
                "msg_pred_q_mean": pred_mean,
            }

            return loss, logs

        return _loss_fn


    def cql_regularizer(agent_state, rng, batch):
        rng_random, rng_pi, rng_next = jax.random.split(rng, 3)

        pi = actor_apply_fn(agent_state.actor.params, batch.obs)
        pi_next = actor_apply_fn(agent_state.actor.params, batch.next_obs)

        pi_actions, _ = pi.sample_and_log_prob(seed=rng_pi)
        pi_next_actions, _ = pi_next.sample_and_log_prob(seed=rng_next)


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

            logs = {
                "cql_ood_q_mean": q_ood.mean(),
                "cql_q_pred_mean": q_pred.mean(),
                "cql_q_pi_mean": q_pi.mean(),
                "cql_q_pi_next_mean": q_pi_next.mean(),
                "cql_q_random_mean": q_random.mean(),
            }

            return min_q_loss, logs

        return _loss_fn


    loss_dict = {
            "none": noop_loss,
            "pbrl": lambda x, y, z: pbrl_regularizer(x, y, z, filtered=False,
                                                     use_next_states=False),
            "filtered_pbrl": lambda x, y, z: pbrl_regularizer(x, y, z, filtered=True,
                                                              use_next_states=False),
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
