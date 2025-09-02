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
        args.ensemble_regularizer.

        Every loss should follow the signature:
            loss_name(agent_state, rng, batch):
               and return a loss that can be differentiated + traced

        three level hierarchy:
            1) Factory picks right function when make_train_step is invoked in training
            2) During every train step we specialize based on current state
            (sample actions from curr actor outside grad, etc.)
            3) Innermost loss is the differentiable regularizer called in q_loss_fn
    """

    """
        Regularizers
    """

    def noop_loss(agent_state, rng, batch):
        """
        No-op loss, used when args.ensemble_regularizer is 'none'
        """
        def _noop_loss_fn(critic_params, rng, batch):
            return jnp.array(0.0)

        return _noop_loss_fn
    
    def std_loss(agent_state, rng, batch):
        """ 
            Sample std regularization in function space
            1) Sample a ~ pi(s) and pi(s')
            2) Compute std over Q(s,a) and Q(s',a')
            3) Maximize 

        """
        rng, rng_actor = jax.random.split(rng, 2)
        rng_curr, rng_next = jax.random.split(rng_actor, 2)

        # Sample one action in each s and s_next, no grad 
        pi_actions = actor_apply_fn(agent_state.actor.params, batch.obs).sample(seed=rng_curr)
        pi_next_actions = actor_apply_fn(agent_state.actor.params, batch.next_obs).sample(seed=rng_next)

        def _loss_fn(critic_params, rng, batch):

            q_values_curr = q_apply_fn(critic_params, batch.obs, pi_actions)
            q_values_next = q_apply_fn(critic_params, batch.next_obs, pi_next_actions)

            # std across ensemble
            std_loss = -1 * (jnp.std(q_values_curr, axis=1) +
                             jnp.std(q_values_next, axis=1))
            return std_loss.mean()
        return _loss_fn



    def mean_vector_loss(agent_state, rng, batch):
        """ 
            Std regularization in parameter space
            1) Flatten parameters for each ensemble member
            2) Compute mean parameter vector
            3) Maximize l2 norm from mean vector
        """
        def _loss_fn(critic_params, rng, batch):
            # flatten parameter pytree
            params, _ = jax.tree_util.tree_flatten(critic_params)
            # Reshape to (E,X) where X is number of params in this kernel / bias
            flat_params = [jnp.reshape(p, (p.shape[0], -1)) for p in params]
            concat_params = jnp.concatenate(flat_params, axis=1)
            # calculate theta_i - theta_mean for each ensemble member
            deviation = concat_params - jnp.mean(concat_params, axis=0, keepdims=True)
            # (E,) vector of L2 deviation norms
            loss = -jnp.linalg.norm(deviation, axis=1)
            return loss.mean()
        return _loss_fn


    def edac_loss(agent_state, rng, batch):
        """
            Edac diversity loss
            1) Calculate dQ(s,a)/da for each ensemble member and (s,a) pair in batch
            2) Compute pairwise dot products
            3) Penalize similarity
        """

        def _loss_fn(critic_params, rng, batch):
            def _diversity_loss_fn(obs, action):
                # shape (E, A) ensemble outputs, A inputs (action)
                action_jac = jax.jacrev(q_apply_fn, argnums=2)(critic_params, obs, action)
                # shape (E,A), normalized gradients for each ensemble member
                action_jac /= jnp.linalg.norm(action_jac, axis=-1, keepdims=True) + 1e-6
                # shape (E,E) pairwise diversity loss
                div_loss = action_jac @ action_jac.T
                # Mask diagonal 
                div_loss *= 1.0 - jnp.eye(args.num_critics)
                return div_loss.sum()

            # vmap over whole batch
            diversity_loss = jax.vmap(_diversity_loss_fn)(batch.obs, batch.action)
            return diversity_loss.mean()

        return _loss_fn

    
    """
        Registering losses, lookup
    """

    loss_dict = {
            "none" : noop_loss,
            "std": std_loss,
            "mean_vector": mean_vector_loss,
            "edac": edac_loss
    }

    if args.ensemble_regularizer in loss_dict:
        print(f"Using ensemble regularizer: {args.ensemble_regularizer}")
        return loss_dict[args.ensemble_regularizer]

    else:
        raise ValueError(f"Unknown ensemble regularizer: {args.ensemble_regularizer}. "
                         f"Available regularizers: {list(loss_dict.keys())}.")



def select_regularizer(args, actor_apply_fn, q_apply_fn):
    """
        Helper function to select regularizer
    """
    return regularizer_factory(args, actor_apply_fn, q_apply_fn)

