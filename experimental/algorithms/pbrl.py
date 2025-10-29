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


from infra import make_pretrain_step, select_regularizer
from infra.dataset import OfflineDatasetWrapper
from infra.utils import linear_schedule, constant_schedule, exponential_schedule, combined_schedule


os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

"""
    Checkpointing
"""

def create_checkpoint_dir():
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{args.algorithm}_{args.dataset_name.replace('/', '.')}/{time_str}"
    ckpt_dir = os.path.join(args.checkpoint_dir, dir_name)
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save args to JSON inside the checkpoint dir
    args_path = os.path.join(ckpt_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(asdict(args), f, indent=2)

    return ckpt_dir

def save_train_state(train_state, ckpt_dir, step):
    checkpoints.save_checkpoint(ckpt_dir, target=train_state, step=step,
                                overwrite=False, keep=2)
    print(f"Checkpoint saved at step {step} in {ckpt_dir}")


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset_source : str = "d4rl"
    dataset_name: str = "halfcheetah-medium-v2"
    algorithm: str = "pbrl"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    checkpoint : bool = False
    checkpoint_dir: str = "./checkpoints"
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
    # --- PBRL --- 
    critic_norm: str = "none" # \in {"spectral", "layer", "none"}
    actor_lr: float = 1e-4
    beta_id : float = 0.01
    beta_ood_start: float = 5.0
    beta_ood_min: float = 0.2
    beta_linear_decay_steps: int = 50_000 # When to switch from linear to exp decay
    beta_ood_exp_decay = 1.01 # Exp decay factor
    constant_beta_ood: bool = False # If enabled, we just keep beta_ood as constant
    ood_actions_sampled: int = 10
    use_next_states : bool = False
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


class AugmentedTrainState(TrainState):
    batch_stats: any = None

AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha pretrain_lag train_step")
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

"""
    Models
"""

class SoftQNetwork(nn.Module):
    depth: int = 3
    critic_norm: str = "none"
    learnable: bool = True
    @nn.compact
    def __call__(self, obs, action, train=True):
        x = jnp.concatenate([obs, action], axis=-1)

        for _ in range(self.depth):
            layer = nn.Dense(256, bias_init=constant(0.1))

            if self.learnable:
                # Normalization for learnable Q-nets
                if self.critic_norm == "spectral":
                    x = nn.SpectralNorm(layer)(x, update_stats=train)
                elif self.critic_norm == "layer":
                    x = layer(x)
                    x = nn.LayerNorm()(x)
                else:
                    # no normalization
                    x = layer(x)
            else:
                # Non-learnable nets (prior) have no normalization
                x = layer(x)

            x = nn.relu(x)

        # For learnable Q-nets, we use a different last layer init
        if self.learnable:
            last_layer = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))
        else:
            last_layer = nn.Dense(1, kernel_init=he_normal, bias_init=sym(3e-3))
        q = last_layer(x)

        return q.squeeze(-1)


class RandomizedPriorQNetwork(nn.Module):
    depth: int 
    scale: float  
    @nn.compact
    def __call__(self, obs, action, train=True):
        q_learnable = SoftQNetwork(learnable=True, critic_norm=args.critic_norm,
                                   name="learnable_q_network")(obs, action, train)
        prior_net = SoftQNetwork(learnable=False, critic_norm="none",
                                 depth=self.depth, name="prior_q_network")
        q_prior = prior_net(obs, action)
        # make sure to not prop grad thru prior net
        q_prior = jax.lax.stop_gradient(q_prior)
        # Combine learnable and prior
        return q_learnable + self.scale * q_prior


class VectorQ(nn.Module):
    num_critics: int
    @nn.compact
    def __call__(self, obs, action, train=True):
        vmap_critic = nn.vmap(
                partial(SoftQNetwork, critic_norm=args.critic_norm, learnable=True), # all learnable
                variable_axes={"params": 0, "batch_stats" : 0},  # Parameters not shared between critics
                split_rngs={"params": True, "dropout": True},  # Different initializations
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action, train)
        return q_values


class PriorVectorQ(nn.Module):
    num_critics: int
    depth: int
    scale: float
    @nn.compact
    def __call__(self, obs, action, train=True):
        vmap_critic = nn.vmap(
                RandomizedPriorQNetwork,
                variable_axes={"params": 0, "batch_stats" : 0},  # Parameters not shared between critics
                split_rngs={"params": True, "dropout": True},  # Different initializations
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_critics,
                )
        q_values = vmap_critic(depth=self.depth, scale=self.scale)(obs, action, train)
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


def create_aug_train_state(args, rng, network, dummy_input, lr=None):
    variables = network.init(rng, *dummy_input)
    return AugmentedTrainState.create(
            apply_fn=network.apply,
            params=variables.get("params"),
            batch_stats = variables.get("batch_stats", None),
            tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
            )

def create_train_state(args, rng, network, dummy_input, lr=None):
    return AugmentedTrainState.create(
            apply_fn=network.apply,
            params = network.init(rng, *dummy_input),
            tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
            )

"""
    Wrapper around Q net apply to handle spectral normalization stats

        Implicitly pass train=true
        If not using any batch statistics, we simply pass empty dictionaries.
"""


def make_q_apply_fn(q_net):
    def q_apply_fn(params, batch_stats, obs, action, train=True):

        # Handle empty dicts when no normalization state is used
        no_normalization = batch_stats is None

        # Need to pass empty dicts if no norm
        # But return same type (None) at the end
        passed_stats = {} if no_normalization else batch_stats

        if train:
            outputs, new_state = q_net.apply(
                {"params": params, "batch_stats": passed_stats},
                obs,
                action,
                train=True,
                mutable=["batch_stats"]
            )

            if no_normalization:
                return outputs, None

            return outputs, new_state["batch_stats"]

        else:
            outputs = q_net.apply(
                {"params": params, "batch_stats": passed_stats},
                obs,
                action,
                train=False,
                mutable=False
            )

            if no_normalization:
                return outputs, None

            return outputs, passed_stats  # just reuse the current batch_stats

    return q_apply_fn



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


def make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset):
    """
    Make JIT-compatible agent train step.

        Setup scheduling and regularizer fns.

        Ensemble regularizer (diversity)
            If set to "none", ensemble regularizer fn will be a no-op.
    """
    ensemble_regularizer_fn = select_regularizer(args, actor_apply_fn, q_apply_fn)

    """
        OOD-regularization

        We stick to PBRL-s two stage schedule for beta_ood
    """
    assert args.constant_beta_ood or args.beta_ood_start >= 1.0 
    assert args.critic_norm in {"spectral", "layer", "none"}
    assert args.beta_ood_min < 1.0
    assert args.constant_beta_ood or args.beta_linear_decay_steps < args.num_updates

    if args.constant_beta_ood:
        # PBRL uses this for expert datasets :-)
        schedule_fn = constant_schedule(args.beta_ood_start)

    else:
        """
            Scheduling for beta_ood

                First decay linearly to 1.0
        """
        linear_schedule_fn = linear_schedule(start=args.beta_ood_start, end=1.0, max_steps=args.beta_linear_decay_steps)

        """
            Then every epoch decay exponentially (/1.01) up to min value
                PBRL actually maintains a running mean of OOD q values and only
                reduces this when the mean decreases in the last epochs, but we
                simplify this to fixed exponential decay.
        """
        exponential_schedule_fn = exponential_schedule(start=1.0,
                                                       min_value=args.beta_ood_min,
                                                       decay_rate=args.beta_ood_exp_decay,
                                                       decay_steps=1000,
                                                       offset=args.beta_linear_decay_steps)

        """
            Create combined schedule
        """
        schedules = [linear_schedule_fn, exponential_schedule_fn]
        steps = [0, args.beta_linear_decay_steps]
        schedule_fn = combined_schedule(schedules, steps)

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Get scheduled hyperparams ---
        step = agent_state.train_step
        beta_ood = schedule_fn(step)

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

        # --- Compute OOD actions, etc. for whole batch ---
        rng_next, rng_ood_next, rng_ood = jax.random.split(rng, 3)

        # --- Get policies for states ---
        next_pis = actor_apply_fn(agent_state.actor.params, batch.next_obs)
        pis = actor_apply_fn(agent_state.actor.params, batch.obs)

        # --- Sample actions ---
        bootstrap_actions, logprobs_next = next_pis.sample_and_log_prob(seed=rng_next)
        logprobs_next = logprobs_next.sum(-1, keepdims=True)
        ood_actions, _ = pis.sample_and_log_prob(seed=rng_ood,
                                                 sample_shape=(args.ood_actions_sampled))
        ood_actions_next, _ = next_pis.sample_and_log_prob(seed=rng_ood_next,
                                                           sample_shape=(args.ood_actions_sampled))

        # Reshape to [B, sampled_actions, action_dim] 
        ood_actions = jnp.swapaxes(ood_actions, 0, 1)
        ood_actions_next = jnp.swapaxes(ood_actions_next, 0, 1)

        # Expand state tensor to [B, sampled_actions, state_dim]
        states = jnp.repeat(jnp.expand_dims(batch.obs, 1), args.ood_actions_sampled, axis=1)
        next_states = jnp.repeat(jnp.expand_dims(batch.next_obs, 1), args.ood_actions_sampled, axis=1)

        # --- Get values ----
        # --- Bootstrap actions with target nets ---
        next_q, _ = q_apply_fn(agent_state.vec_q_target.params, 
                               agent_state.vec_q_target.batch_stats,
                               batch.next_obs, bootstrap_actions, train=False)


        # --- OOD actions with current Q nets --- 
        ood_q, new_bs = q_apply_fn(agent_state.vec_q.params, 
                                   agent_state.vec_q.batch_stats,
                                   states, ood_actions)

        ood_q_next, new_bs = q_apply_fn(agent_state.vec_q.params,
                                        new_bs,
                                        next_states, ood_actions_next)


        agent_state = agent_state._replace(vec_q=agent_state.vec_q.replace(batch_stats=new_bs))

        # [B, 1]
        std_q_next = next_q.std(-1, keepdims=True)
        # [B, sampled_actions, 1]
        std_q_ood = ood_q.std(-1, keepdims=True)
        std_q_ood_next = ood_q_next.std(-1, keepdims=True)

        next_v_target = next_q - alpha * logprobs_next - args.beta_id * std_q_next

        """
            OOD targets
        """
        ood_target_curr = ood_q - beta_ood * std_q_ood

        """
            PBRL keeps ood penalty for next state actions
            as fixed 0.1 for some reason.
        """
        old_target_next = ood_q_next - 0.1 * std_q_ood_next

        """
            Clip targets to be non-negative for stability (!This is incorrect for negative rewards!)
        """
        ood_target_curr = jnp.clip(ood_target_curr, a_min=0.0)
        old_target_next = jnp.clip(old_target_next, a_min=0.0)

        """
            TD target
        """
        td_target = jnp.expand_dims(batch.reward, -1) + args.gamma * jnp.expand_dims((1 - batch.done), -1) * next_v_target

        # --- Get specialized regularizer loss function with current state --- 
        rng, rng_reg, rng_reg_loss = jax.random.split(rng, 3)
        ensemble_reg_loss = ensemble_regularizer_fn(agent_state, rng_reg, batch)

        # --- Update critics ---
        @partial(jax.value_and_grad, has_aux=True)
        def _q_loss_fn(params):

            # [B, ensemble_size]
            q_pred, new_bs = q_apply_fn(params, 
                                        agent_state.vec_q.batch_stats,
                                        batch.obs, batch.action)

            # Bellman error
            critic_loss = jnp.square((q_pred - td_target))
            # Take mean over batch and sum over ensembles
            critic_loss = critic_loss.sum(-1).mean()

            """
                OOD regularization
            """
            # Q(s,a) for a ~ pi(s), shape [B * ood_actions_sampled, ensemble_size]
            q_pred_ood, new_bs = q_apply_fn(params, 
                                            new_bs,
                                            states, ood_actions)

            q_pred_ood_next, new_bs = q_apply_fn(params, 
                                                 new_bs,
                                                 next_states, ood_actions_next)

            # on current states
            ood_loss = jnp.square(q_pred_ood - ood_target_curr).sum(-1).mean()
            # on successor states
            ood_loss += jnp.square(q_pred_ood_next - old_target_next).sum(-1).mean()
            critic_loss += ood_loss

            """
                Ensemble regularizer
            """
            regularizer_loss = ensemble_reg_loss(params, rng_reg_loss, batch)
            critic_loss += args.reg_lagrangian * regularizer_loss

            return critic_loss, (new_bs, ood_loss, regularizer_loss, q_pred.mean(), \
                    q_pred.std(), q_pred_ood.mean(), q_pred_ood.std())

        (critic_loss, (new_bs, ood_loss,regularizer_loss, q_pred_mean, q_pred_std,
                       q_pred_ood_mean, q_pred_ood_std)), critic_grad = _q_loss_fn(agent_state.vec_q.params)

        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad, 
                                                      batch_stats=new_bs)
        # Update state
        agent_state = agent_state._replace(vec_q=updated_q)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
                agent_state.vec_q.params,
                agent_state.vec_q_target.params,
                args.polyak_step_size,
                )
        updated_q_target = agent_state.vec_q_target.replace(
                step=agent_state.vec_q_target.step + 1,
                params=updated_q_target_params,
                batch_stats=agent_state.vec_q.batch_stats  # copy batch_stats
                )
        agent_state = agent_state._replace(vec_q_target=updated_q_target)

        # --- Update actor ---
        @partial(jax.value_and_grad, has_aux=True)
        def _actor_loss_function(params, rng):
            def _compute_loss(rng, transition):
                pi = actor_apply_fn(params, transition.obs)
                sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                log_pi = log_pi.sum()
                q_values, new_bs = q_apply_fn(
                        agent_state.vec_q.params,
                        agent_state.vec_q.batch_stats,
                        transition.obs, sampled_action
                        )

                std_q = q_values.std(-1)
                # Use minimum as PI operator
                q_tgt = q_values.min(-1)
                return -q_tgt + alpha * log_pi, -log_pi, q_tgt, std_q

            rng = jax.random.split(rng, args.batch_size)
            loss, entropy, q_lcb, q_std = jax.vmap(_compute_loss)(rng, batch)
            # compute mean q-value and mean std over actions
            return loss.mean(), (new_bs, entropy.mean(), q_lcb.mean(), q_std.mean())

        rng, rng_actor = jax.random.split(rng)
        (actor_loss, (new_bs, entropy, q_lcb, q_std)), actor_grad = (
                _actor_loss_function(agent_state.actor.params, rng_actor)
                )
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)

        # --- Update actor and batch statistics again ---
        agent_state = agent_state._replace(actor=updated_actor, 
                                          vec_q=agent_state.vec_q.replace(batch_stats=new_bs))

        # --- Increment training step ---
        agent_state = agent_state._replace(train_step=agent_state.train_step + 1)
       

        loss = {
                "critic_loss": critic_loss,
                "beta_ood": beta_ood,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "regularizer_loss": regularizer_loss,
                "ood_loss": ood_loss,
                "entropy": entropy,
                "alpha": alpha,
                "actor_q_lcb": q_lcb,
                "q_pred_mean" : q_pred_mean,
                "q_pred_std": q_pred_std,
                "q_pred_ood_mean": q_pred_ood_mean,
                "q_pred_ood_std": q_pred_ood_std,
                }

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
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs], args.actor_lr),
        vec_q=create_aug_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        vec_q_target=create_aug_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        alpha=create_train_state(args, rng_alpha, alpha_net, []),
        pretrain_lag=jnp.full((), args.pretrain_lag_init, dtype=jnp.float32),
        train_step=jnp.full((), 0, dtype=jnp.float32),
    )

    # --- Wrap Q apply fn ---
    q_apply = make_q_apply_fn(q_net)

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        args, actor_net.apply, q_apply, alpha_net.apply, dataset
    )

    # --- Make pretrain step ---
    _agent_pretrain_step_fn = make_pretrain_step(
        args, actor_net.apply, q_apply, alpha_net.apply, dataset
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
            print("Actor loss: ", loss["actor_loss"][-1])
            print("Critic loss: ", loss["critic_loss"][-1])

            if args.log:
                log_dict = {
                    "return": returns.mean(),
                    "score": scores.mean(),
                    "score_std": scores.std(),
                    "num_updates": step,
                    **{k: loss[k][-1] for k in loss},
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
        if args.log:
            log_dict = {
                "return": returns.mean(),
                "score": scores.mean(),
                "score_std": scores.std(),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
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
        os.makedirs(f"final_returns/{args.algorithm}/", exist_ok=True)
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
