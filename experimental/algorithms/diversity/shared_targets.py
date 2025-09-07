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
from flax.training import checkpoints
from flax.training.train_state import TrainState
import gym
import jax
import jax.numpy as jnp
import json
import numpy as onp
import optax
import tyro
import wandb

from infra.ensemble_regularization import select_regularizer
from infra.pretraining import make_pretrain_step
from infra.diversity_utils import diversity_loss, prepare_ood_dataset, \
    compute_qvalue_statistics, get_diversity_statistics
from infra.offline_dataset_wrapper import OfflineDatasetWrapper

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

"""
    Checkpointing
"""

def create_checkpoint_dir():
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{args.algorithm}_{args.dataset_name.replace('/', '.')}/{time_str}"
    ckpt_dir = os.path.join("./checkpoints", dir_name)
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save args to JSON inside the checkpoint dir
    args_path = os.path.join(ckpt_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(asdict(args), f, indent=2)

    return ckpt_dir

def save_train_state(train_state, ckpt_dir, step):
    checkpoints.save_checkpoint(ckpt_dir, target=train_state, step=step)
    print(f"Checkpoint saved at step {step} in {ckpt_dir}")


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset_source : str = "d4rl"
    dataset_name: str = "walker2d-medium-v2"
    algorithm: str = "sac_n"
    num_updates: int = 3_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    checkpoint : bool = False
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
    # ---  Pretraining ---
    pretrain_updates : int = 0
    pretrain_loss : str = "bc+sarsa"
    pretrain_lag_init: float = 1.0
    # --- Diversity --- 
    ensemble_regularizer : str = "none"
    reg_lagrangian: float = 1.0
    # --- RPF --- 
    prior: bool = False
    randomized_prior_depth : int = 3
    randomized_prior_scale : float = 1.0


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha pretrain_lag")
Transition = namedtuple("Transition", "obs action reward next_obs next_action done")


"""
    Initializers
"""
def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init

he_normal = nn.initializers.variance_scaling(
    scale=2.0,
    mode="fan_in",
    distribution="truncated_normal" 
)

class SoftQNetwork(nn.Module):
    depth: int = 3
    learnable: bool = True
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        if self.learnable:
            # For learnable Q-nets, we use a different last layer init
            q = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))(x)
        else:
            q = nn.Dense(1, kernel_init=he_normal, bias_init=sym(3e-3))(x)

        return q.squeeze(-1)

class RandomizedPriorQNetwork(nn.Module):
    depth: int 
    scale: float  
    @nn.compact
    def __call__(self, obs, action):
        q_learnable = SoftQNetwork(learnable=True, name="learnable_q_network")(obs, action)
        prior_net = SoftQNetwork(learnable=False, depth=self.depth, name="prior_q_network")
        q_prior = prior_net(obs, action)
        # make sure to not prop grad thru prior net
        q_prior = jax.lax.stop_gradient(q_prior)
        # Combine learnable and prior
        return q_learnable + self.scale * q_prior


class VectorQ(nn.Module):
    num_critics: int
    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action)
        return q_values

class PriorVectorQ(nn.Module):
    num_critics: int
    depth: int
    scale: float
    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            RandomizedPriorQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.num_critics,
        )
        q_values = vmap_critic(depth=self.depth, scale=self.scale)(obs, action)
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
        mean = nn.Dense(self.num_actions, kernel_init=sym(1e-3), bias_init=sym(1e-3))(x)
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


def create_train_state(args, rng, network, dummy_input):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(args.lr, eps=1e-5),
    )


def eval_agent(args, rng, env, agent_state):
    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)
    obs = env.reset()

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

    # Select regularizer based on args
    ensemble_regularizer_fn = select_regularizer(args, actor_apply_fn, q_apply_fn)

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
                q_min = jnp.min(q_values)
                return -q_min + alpha * log_pi, -log_pi, q_min, q_values.std()

            rng = jax.random.split(rng, args.batch_size)
            loss, entropy, q_min, q_std = jax.vmap(_compute_loss)(rng, batch)
            return loss.mean(), (entropy.mean(), q_min.mean(), q_std.mean())

        rng, rng_actor = jax.random.split(rng)
        (actor_loss, (entropy, q_pred_min, q_pred_std)), actor_grad = _actor_loss_function(
            agent_state.actor.params, rng_actor
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
            step=agent_state.vec_q_target.step + 1, params=updated_q_target_params
        )
        agent_state = agent_state._replace(vec_q_target=updated_q_target)

        # --- Compute targets ---
        def _sample_next_v(rng, transition):
            next_pi = actor_apply_fn(agent_state.actor.params, transition.next_obs)
            # Note: Important to use sample_and_log_prob here for numerical stability
            # See https://github.com/deepmind/distrax/issues/7
            next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng)
            # Minimum of the target Q-values
            next_q = q_apply_fn(
                agent_state.vec_q_target.params, transition.next_obs, next_action
            )
            return next_q.min(-1) - alpha * log_next_pi.sum(-1)

        rng, rng_next_v = jax.random.split(rng)
        rng_next_v = jax.random.split(rng_next_v, args.batch_size)
        next_v_target = jax.vmap(_sample_next_v)(rng_next_v, batch)
        target = batch.reward + args.gamma * (1 - batch.done) * next_v_target

        rng, rng_reg, rng_reg_loss = jax.random.split(rng, 3)
        # --- Get specialized loss function with current state --- 
        ensemble_reg_loss = ensemble_regularizer_fn(agent_state, rng_reg, batch)

        # --- Update critics ---
        @partial(jax.value_and_grad, has_aux=True)
        def _q_loss_fn(params):
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            critic_loss = jnp.square((q_pred - jnp.expand_dims(target, -1))).sum(-1).mean()
            regularizer_loss = ensemble_reg_loss(params, rng_reg_loss, batch)
            critic_loss += args.reg_lagrangian * regularizer_loss
            return critic_loss, (regularizer_loss, q_pred.mean())

        # --- DIVERSITY: calculate EDAC loss --- 
        diversity_loss_val = diversity_loss(q_apply_fn, agent_state, batch.obs, batch.action, args.num_critics)
        # --- DIVERSITY: get ensemble stats for logging ---
        rng_perturb, rng = jax.random.split(rng)
        diversity_stats = get_diversity_statistics(q_apply_fn, actor_apply_fn,
                                                   agent_state, rng_perturb,
                                                   batch.obs, batch.action)

        (critic_loss, (regularizer_loss, q_pred_mean)), critic_grad = _q_loss_fn(agent_state.vec_q.params)
        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
        agent_state = agent_state._replace(vec_q=updated_q)


        loss = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "regularizer_loss": regularizer_loss,
            "diversity_loss": diversity_loss_val,
            "entropy": entropy,
            "alpha": alpha,
        }

        # --- DIVERSITY: add info to logs
        for k, v in diversity_stats.items():
            loss[k] = v

        return (rng, agent_state), loss

    return _train_step


def train(args):
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

    dataset_wrapper = OfflineDatasetWrapper(source=args.dataset_source,
                                            dataset=args.dataset_name)

    # --- DIVERSITY: Load expert dataset for walker
    if args.dataset_name.split("-")[0] != "walker2d":
        print("This is a mock used to test walker2d, don't use a different dataset")
    rng_data, rng = jax.random.split(rng)
    ood_obs, ood_actions = prepare_ood_dataset(
        rng_data, dataset_name="walker2d-expert-v2", ood_samples=50
    )

    # --- Initialize environment and dataset ---
    rng, rng_env = jax.random.split(rng)
    env = dataset_wrapper.get_eval_env(args.eval_workers, rng_env)

    dataset = dataset_wrapper.get_dataset()
    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        next_action=jnp.roll(jnp.array(dataset["actions"]), -1, axis=0),
        done=jnp.array(dataset["terminals"]),
    )

    # --- Initialize agent and value networks ---
    num_actions = env.single_action_space.shape[0]
    dummy_obs = jnp.zeros(env.single_observation_space.shape)
    dummy_action = jnp.zeros(num_actions)
    actor_net = TanhGaussianActor(num_actions)

    # --- Init Q, include prior net if enabled ---
    if args.prior:
        q_net = PriorVectorQ(
            num_critics=args.num_critics,
            depth=args.randomized_prior_depth,
            scale=args.randomized_prior_scale,
        )
    else:
        q_net = VectorQ(args.num_critics)

    alpha_net = EntropyCoef()

    # Target networks share seeds to match initialization
    rng, rng_actor, rng_q, rng_alpha, rng_lag = jax.random.split(rng, 5)
    agent_state = AgentTrainState(
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        vec_q=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        vec_q_target=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        alpha=create_train_state(args, rng_alpha, alpha_net, []),
        pretrain_lag=jnp.full((), args.pretrain_lag_init, dtype=jnp.float32),
    )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        args, actor_net.apply, q_net.apply, alpha_net.apply, dataset
    )

    # --- Make pretrain step ---
    _agent_pretrain_step_fn = make_pretrain_step(
        args, actor_net.apply, q_net.apply, alpha_net.apply, dataset
    )

    """
        Pretraining
    """

    assert(args.pretrain_updates <= args.num_updates), \
            "pretrain_updates must be less than or equal to total updates"

    pretrain_evals = args.pretrain_updates // args.eval_interval

    if args.pretrain_updates > 0:
        for eval_idx in range(pretrain_evals):

            (rng, agent_state), loss = jax.lax.scan(
                _agent_pretrain_step_fn,
                (rng, agent_state),
                None,
                args.eval_interval,
            )
            # --- Evaluate agent ---
            rng, rng_eval = jax.random.split(rng)
            # Evaluates on env from get_eval_env
            returns = dataset_wrapper.eval_agent(args, rng_eval, agent_state)
            scores = dataset_wrapper.get_normalized_score(returns) * 100.0

            # --- Log metrics ---
            step = (eval_idx + 1) * args.eval_interval
            print("Step:", step, f"\t Score: {scores.mean():.2f}")

            # --- DIVERSITY: get info on OOD data ---
            ood_stats = compute_qvalue_statistics(q_net.apply,
                                                  agent_state,
                                                  ood_obs, 
                                                  ood_actions)
            if args.log:
                log_dict = {
                    "return": returns.mean(),
                    "score": scores.mean(),
                    "score_std": scores.std(),
                    "num_updates": step,
                    **{k: loss[k][-1] for k in loss},
                    "ood_q_std": ood_stats["std"],
                    "ood_q_mean": ood_stats["mean"],
                    "ood_q_min": ood_stats["min"],
                }
                wandb.log(log_dict)

    if args.checkpoint:
        ckpt_dir = create_checkpoint_dir()
        save_train_state(agent_state, ckpt_dir, pretrain_evals)

    num_evals = (args.num_updates - args.pretrain_updates) // args.eval_interval

    """
        Offline Training
    """

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
        returns = dataset_wrapper.eval_agent(args, rng_eval, agent_state)
        scores = dataset_wrapper.get_normalized_score(returns) * 100.0

        # --- Log metrics ---
        step = (eval_idx + 1 + pretrain_evals) * args.eval_interval
        print("Step:", step, f"\t Score: {scores.mean():.2f}")

        # --- DIVERSITY: get info on OOD data ---
        ood_stats = compute_qvalue_statistics(q_net.apply,
                                              agent_state,
                                              ood_obs, 
                                              ood_actions)
        if args.log:
            log_dict = {
                "return": returns.mean(),
                "score": scores.mean(),
                "score_std": scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
                "ood_q_std": ood_stats["std"],
                "ood_q_mean": ood_stats["mean"],
                "ood_q_min": ood_stats["min"],
            }
            wandb.log(log_dict)

    if args.checkpoint:
        save_train_state(agent_state, ckpt_dir, num_evals + pretrain_evals)

    # --- Evaluate final agent ---
    if args.eval_final_episodes > 0:
        final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng = jax.random.split(rng, final_iters)
        rets = onp.concatenate([dataset_wrapper.eval_agent(args, _rng, agent_state) for _rng in _rng])
        print("Returns: ", rets)
        env.close()
        scores = dataset_wrapper.get_normalized_score(rets) * 100.0
        agg_fn = lambda x, k: {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}
        info = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")

        # --- Write final returns to file ---
        os.makedirs("final_returns", exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filtered_name = args.dataset_name.replace("/", "_").replace("-", "_")

        filename = f"{args.algorithm}_{filtered_name}_{time_str}.npz"
        with open(os.path.join("final_returns", filename), "wb") as f:
            onp.savez_compressed(f, **info, args=asdict(args))

        if args.log:
            wandb.save(os.path.join("final_returns", filename))

    if args.log:
        wandb.finish()



if __name__ == "__main__":
    # --- Parse arguments ---
    args = tyro.cli(Args)
    # --- Train agent ---
    train(args)
