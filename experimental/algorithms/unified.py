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

from infra.models.actor import TanhGaussianActor, EntropyCoef
from infra.models.critic import VectorQ, PriorVectorQ
from infra.models.normalization_wrapper import NormalizationWrapper


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

    # --- Environment --- 
    action_scale: float = 1.0 # Scale action space from [-1, 1]^d to [-scale, scale]^d

    # --- Generic optimization ---
    actor_lr: float = 1e-4
    lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    polyak_step_size: float = 0.005


    # --- SAC-N ---
    num_critics: int = 10

    # --- Policy Evaluation ---
    shared_targets : bool = False
    beta_id : float = 0.01 # If independent targets, controls std penalty (target - beta_id * std)

    # --- Policy Improvement ---
    pi_operator: str = "min" # \in {"min", lcb"}
    actor_lcb_penalty: float = 4.0 # Used if operator is lcb to penalize with std

    # --- Critic Regularization --- 
    critic_regularizer: str = "none" # \in {"none", "cql", "pbrl", "msg"}
    critic_lagrangian: float = 1.0
    critic_norm: str = "none" # \in {"none", "spectral", "layer"}
    critic_regularizer_parameter : int = 1 # Num of sampled actions for PBRL, temp for CQL

    # ---  Pretraining ---
    pretrain_updates : int = 0
    pretrain_loss : str = "bc+sarsa"
    pretrain_lagrangian: float = 1.0

    # --- Diversity Regularization --- 
    ensemble_regularizer : str = "none"
    reg_lagrangian: float = 1.0

    # --- RPF --- 
    prior: bool = False
    randomized_prior_depth : int = 3
    randomized_prior_scale : float = 1.0


"""
    Augmented Train state to use normalization
"""

class AugmentedTrainState(TrainState):
    batch_stats: any = None

AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha pretrain_lag train_step")
Transition = namedtuple("Transition", "obs action reward next_obs next_action done")

def create_aug_train_state(args, rng, network, dummy_input, lr=None):
    variables = network.init(rng, *dummy_input)
    return AugmentedTrainState.create(
            apply_fn=network.apply,
            params=variables.get("params"),
            batch_stats = variables.get("batch_stats", {}),
            tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
            )

def create_train_state(args, rng, network, dummy_input, lr=None):
    return AugmentedTrainState.create(
            apply_fn=network.apply,
            params = network.init(rng, *dummy_input),
            tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
            )


def make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset):
    """
    Make JIT-compatible agent train step.

        Setup scheduling and regularizer fns.

        Ensemble regularizer (diversity)
            If set to "none", ensemble regularizer fn will be a no-op.
    """
    ensemble_regularizer_fn = select_regularizer(args, actor_apply_fn, q_apply_fn)

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Get scheduled hyperparams ---
        step = agent_state.train_step

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
                rng_batch, (args.batch_size,), 0, len(dataset.obs)
                )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        """
            Update alpha (Entropy temperature)
        """
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

        """
            Compute TD targets
        """
        # --- Sample actions for batch states ---
        rng_next, rng = jax.random.split(rng, 2)
        next_pis = actor_apply_fn(agent_state.actor.params, batch.next_obs)
        bootstrap_actions, logprobs_next = next_pis.sample_and_log_prob(seed=rng_next)
        logprobs_next = logprobs_next.sum(-1, keepdims=True)

        # --- Bootstrap actions with target nets ---
        next_q, _ = q_apply_fn(agent_state.vec_q_target.params, 
                               agent_state.vec_q_target.batch_stats,
                               batch.next_obs, bootstrap_actions, train=False)

        # --- Compute TD target ---
        std_q_next = next_q.std(-1, keepdims=True)
        next_v_target = next_q - alpha * logprobs_next - args.beta_id * std_q_next
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

            ood_loss = jnp.array(0.0)
            q_pred_ood = jnp.array(0.0)

            """
                Ensemble regularizer
            """
            regularizer_loss = ensemble_reg_loss(params, rng_reg_loss, batch)
            critic_loss += args.reg_lagrangian * regularizer_loss

            return critic_loss, (new_bs, ood_loss, regularizer_loss, q_pred.mean(), \
                    q_pred.std(), q_pred_ood.mean(), q_pred_ood.std())

        (critic_loss, (new_bs, ood_loss,regularizer_loss, q_pred_mean, q_pred_std,
                       q_pred_ood_mean, q_pred_ood_std)), critic_grad = _q_loss_fn(agent_state.vec_q.params)


        # --- Update Q-state
        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad, 
                                                      batch_stats=new_bs)
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
        agent_state = agent_state._replace(actor=updated_actor,
                                           vec_q=agent_state.vec_q.replace(batch_stats=new_bs))
        agent_state = agent_state._replace(train_step=agent_state.train_step + 1)

        loss = {
                    "critic_loss": critic_loss,
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
            critic_norm=args.critic_norm,
            depth=args.randomized_prior_depth,
            scale=args.randomized_prior_scale,
        )
    else:
        q_net = VectorQ(args.num_critics, critic_norm=args.critic_norm)

    alpha_net = EntropyCoef()

    # Target networks share seeds to match initialization
    rng, rng_actor, rng_q, rng_alpha, rng_lag = jax.random.split(rng, 5)
    agent_state = AgentTrainState(
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs], args.actor_lr),
        vec_q=create_aug_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        vec_q_target=create_aug_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        alpha=create_train_state(args, rng_alpha, alpha_net, []),
        pretrain_lag=jnp.full((), args.pretrain_lagrangian, dtype=jnp.float32),
        train_step=jnp.full((), 0, dtype=jnp.float32),
    )

    # --- Wrap Q apply fn ---
    q_apply = NormalizationWrapper(q_net)

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
