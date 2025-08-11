from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
import os
import warnings

import distrax
import d4rl
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from flax.training.train_state import TrainState
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tyro
import wandb

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "msg"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    # --- Generic optimization ---
    lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    polyak_step_size: float = 0.005
    # --- SAC-N ---
    num_critics: int = 10
    # --- MSG ---
    cql_min_q_weight: float = 0.5
    actor_lcb_coef: float = 4.0


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple(
    "AgentTrainState", "actor vec_q vec_q_target alpha"
)
Transition = namedtuple("Transition", "obs action reward next_obs done")


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class SoftQNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        # [B,S] , [B,A] -> [B,S+A]
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        # Small variance for initialization of last layer
        q = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))(x)
        return q.squeeze(-1)


class VectorQ(nn.Module):
    num_critics: int

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={
                "params": 0
            },  # Parameters not shared between critics
            split_rngs={
                "params": True,
                "dropout": True,
            },  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action)
        return q_values


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


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return log_ent_coef


def create_train_state(args, rng, network, dummy_input, lr=None):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
    )


def eval_agent(args, rng, env, agent_state):
    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)

    def _rng_to_integer_seed(rng):
        return int(jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max))

    seeds_reset = [_rng_to_integer_seed(rng) for rng in rng_reset]
    obs = env.reset(seed=seeds_reset)

    # --- Rollout agent ---
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action)

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        # --- Take step in environment ---
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, done, info = env.step(onp.array(action))

        # --- Track cumulative reward ---
        cum_reward += reward * ~returned
        returned |= done

    if step >= max_episode_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward


r"""
          __/)
       .-(__(=:
    |\ |    \)
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Agent
"""


def make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset):
    """Make JIT-compatible agent train step."""

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
        alpha_loss, alpha_grad = _alpha_loss_fn(
            agent_state.alpha.params, rng_alpha
        )
        updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
        agent_state = agent_state._replace(alpha=updated_alpha)
        alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))

        # --- Update actor ---
        @partial(jax.value_and_grad, has_aux=True)
        def _actor_loss_function(params, rng):
            def _compute_loss(rng, transition):
                pi = actor_apply_fn(params, transition.obs)
                sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                log_pi = log_pi.sum()
                q_values = q_apply_fn(
                    agent_state.vec_q.params, transition.obs, sampled_action
                )
                # calculate LCB over ensemble
                std_q = q_values.std(-1)
                q_lcb = q_values.mean(-1) - args.actor_lcb_coef * std_q

                return -q_lcb + alpha * log_pi, -log_pi, q_lcb, std_q

            rng = jax.random.split(rng, args.batch_size)
            loss, entropy, q_lcb, q_std = jax.vmap(_compute_loss)(rng, batch)
            # compute mean q-value and estimate std over actions
            return loss.mean(), (entropy.mean(), q_lcb.mean(), q_std.mean())

        rng, rng_actor = jax.random.split(rng)
        (actor_loss, (entropy, q_lcb, q_std)), actor_grad = (
            _actor_loss_function(agent_state.actor.params, rng_actor)
        )
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.vec_q.params,
            agent_state.vec_q_target.params,
            args.polyak_step_size,
        )
        updated_q_target = agent_state.vec_q_target.replace(
            step=agent_state.vec_q_target.step + 1,
            params=updated_q_target_params,
        )
        agent_state = agent_state._replace(vec_q_target=updated_q_target)

        # --- Compute targets ---
        def _sample_next_v(rng, transition):
            next_pi = actor_apply_fn(
                agent_state.actor.params, transition.next_obs
            )
            # Note: Important to use sample_and_log_prob here for numerical stability
            # See https://github.com/deepmind/distrax/issues/7
            next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng)

            # Keep independent targets [B, ensemble_size], do not reduce via min(-1)
            next_q = q_apply_fn(
                agent_state.vec_q_target.params,
                transition.next_obs,
                next_action,
            )
            return next_q - alpha * log_next_pi.sum(-1)

        rng, rng_next_v = jax.random.split(rng)
        rng_next_v = jax.random.split(rng_next_v, args.batch_size)
        next_v_target = jax.vmap(_sample_next_v)(rng_next_v, batch)

        # Expand [B] -> [B, 1] and broadcast to get [B, ensemble_size] targets
        target = jnp.expand_dims(batch.reward, -1) + args.gamma * jnp.expand_dims((1 - batch.done), -1) * next_v_target

        # --- Sample actions for CQL ---
        def _sample_actions(rng, obs):
            pi = actor_apply_fn(agent_state.actor.params, obs)
            return pi.sample(seed=rng)

        rng, rng_pi = jax.random.split(rng, 2)
        pi_actions = _sample_actions(rng_pi, batch.obs)

        # --- Update critics ---
        @partial(jax.value_and_grad, has_aux=True)
        def _q_loss_fn(params):
            # [B, ensemble_size]
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            critic_loss = jnp.square((q_pred - target))
            # Take mean over batch and sum over ensembles
            critic_loss = critic_loss.sum(-1).mean()

            # Q(s,a) for a ~ pi(s), shape [B, ensemble_size]
            pi_q = q_apply_fn(params, batch.obs, pi_actions)
            # [B, 1] for each s in batch, reduce over ensemble dim
            q_diff = (pi_q - q_pred).sum(-1)
            min_q_loss = q_diff * args.cql_min_q_weight

            critic_loss += min_q_loss.mean()
            return critic_loss, (q_pred.mean(), q_pred.std(),
                                 pi_q.mean(), pi_q.std())

        # unpack aux and get critic grad
        (critic_loss, (q_pred_mean, q_pred_std, pi_q_mean, pi_q_std)), critic_grad = _q_loss_fn(agent_state.vec_q.params)

        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
        agent_state = agent_state._replace(vec_q=updated_q)

        def get_bias_estimates(rng, params, variances):

            rng, rng_q = jax.random.split(rng)
            rng_q = jax.random.split(rng_q, variances.shape[0])  # Shape: (n,)
            

            # Calculate original Q-values (batch shape: (n, ...))
            q_pred = q_apply_fn(agent_state.vec_q.params, batch.obs, batch.action)


            def _perturb_q_values(rng, obs, actions, noise_variance):
                # Sample noise from [-var, var]
                eps = jax.random.uniform(
                    rng,
                    shape=actions.shape,
                    minval=-noise_variance,
                    maxval=+noise_variance,
                )

                # Perturb and clip actions
                perturbed_action = actions + eps
                perturbed_action = jnp.clip(perturbed_action, -1.0, 1.0)

                perturbed_q = q_apply_fn(
                    params, obs, perturbed_action
                )

                return perturbed_q

            # Broadcast variances to (n, 1) before vmap
            variances = variances[:, None]

            perturbed_q_curr = jax.vmap(
                _perturb_q_values, in_axes=(0, None, None, 0)
            )(rng_q, batch.obs, batch.action, variances)

            # ood penalty (shape (var, ) mean penalty for each coeff)
            penalty = (args.actor_lcb_coef * perturbed_q_curr.std(-1)).mean(-1)

            # calculate Q-gap between perturbed and original Q-values for each critic
            q_gap = jnp.mean(perturbed_q_curr - jnp.expand_dims(q_pred, axis=0), axis=(1,2))
            

            return q_gap, penalty

        num_perturbations = 3
        # Perturb actions from support
        variances = jnp.linspace(0.1, 0.3, num_perturbations) 

        # lol
        variances_py = [0.1, 0.2, 0.3]
        bias_estimates, penalties = get_bias_estimates(rng, agent_state.vec_q.params, variances)
        loss = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "entropy": entropy,
            "alpha": alpha,
            "actor_q_lcb": q_lcb,
            "q_pred_mean" : q_pred_mean,
            "q_pred_std": q_pred_std,
            "pi_q_mean": pi_q_mean,
            "pi_q_std": pi_q_std
}

        # Add pessimism
        for i, var in enumerate(variances_py):
            loss[f"bias_estimate_{var}"] = bias_estimates[i].astype(float)
            loss[f"penalty_{var}"] = penalties[i].astype(float)

        return (rng, agent_state), loss

    return _train_step


if __name__ == "__main__":
    # --- Parse arguments ---
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)

    # --- Initialize logger ---
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type="train_agent",
        )

    # --- Initialize environment and dataset ---
    env = gym.vector.make(args.dataset, num_envs=args.eval_workers)
    dataset = d4rl.qlearning_dataset(gym.make(args.dataset))
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    # --- Initialize agent and value networks ---
    num_actions = env.single_action_space.shape[0]
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(num_actions)
    actor_net = TanhGaussianActor(num_actions)
    q_net = VectorQ(args.num_critics)
    alpha_net = EntropyCoef()

    # Target networks share seeds to match initialization
    rng, rng_actor, rng_q, rng_alpha = jax.random.split(rng, 4)
    agent_state = AgentTrainState(
        actor=create_train_state(
            args, rng_actor, actor_net, [dummy_obs]
        ),
        vec_q=create_train_state(
            args, rng_q, q_net, [dummy_obs, dummy_action]
        ),
        vec_q_target=create_train_state(
            args, rng_q, q_net, [dummy_obs, dummy_action]
        ),
        alpha=create_train_state(args, rng_alpha, alpha_net, []),
    )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        args, actor_net.apply, q_net.apply, alpha_net.apply, dataset
    )

    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state),
            None,
            args.eval_interval,
        )

        # --- Evaluate agent ---
        rng, rng_eval = jax.random.split(rng)
        returns = eval_agent(args, rng_eval, env, agent_state)
        scores = d4rl.get_normalized_score(args.dataset, returns) * 100.0

        # --- Log metrics ---
        step = (eval_idx + 1) * args.eval_interval
        print("Step:", step, f"\t Score: {scores.mean():.2f}")
        if args.log:
            log_dict = {
                "return": returns.mean(),
                "score": scores.mean(),
                "score_std": scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
            }
            wandb.log(log_dict)

    # --- Evaluate final agent ---
    if args.eval_final_episodes > 0:
        final_iters = int(
            onp.ceil(args.eval_final_episodes / args.eval_workers)
        )
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng = jax.random.split(rng, final_iters)
        rets = onp.concatenate(
            [eval_agent(args, _rng, env, agent_state) for _rng in _rng]
        )
        print(rets)
        env.close()
        scores = d4rl.get_normalized_score(args.dataset, rets) * 100.0
        agg_fn = lambda x, k: {
            k: x,
            f"{k}_mean": x.mean(),
            f"{k}_std": x.std(),
        }
        info = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")

        # --- Write final returns to file ---
        os.makedirs("final_returns", exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{args.algorithm}_{args.dataset}_{time_str}.npz"
        with open(os.path.join("final_returns", filename), "wb") as f:
            onp.savez_compressed(f, **info, args=asdict(args))

        if args.log:
            wandb.save(os.path.join("final_returns", filename))

    if args.log:
        wandb.finish()
