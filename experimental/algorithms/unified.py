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
import tqdm

import infra
import infra.utils

from infra import make_pretrain_step, select_regularizer, select_ood_regularizer
from infra.dataset import OfflineDatasetWrapper
from infra.utils import linear_schedule, constant_schedule, \
    exponential_schedule, combined_schedule, print_args

# For diversity logs
from infra.utils.diversity_utils import prepare_ood_dataset, \
        get_diversity_statistics, compute_qvalue_statistics, diversity_loss

from infra.utils.visualization import visualize_q_vals, visualize_reach_bias

from infra.models.actor import TanhGaussianActor, EntropyCoef
from infra.models.critic import VectorQ, PriorVectorQ

from infra.checkpoints import create_checkpoint_dir, get_experiment_dirname, save_train_state



os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset_source : str = "d4rl"
    dataset_name: str = "halfcheetah-medium-v2"
    algorithm: str = "unified"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    checkpoint : bool = False
    checkpoint_dir: str = "./data"
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"

    # --- Environment --- 
    action_scale: float = 1.0 # Scale action space from [-1, 1]^d to [-scale, scale]^d
    reward_scale: float = 1.0 # a
    reward_shift: float = 0.0 # b, rewards transformed r -> a * r + b 

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
    critic_norm: str = "none" # \in {"none", "layer"}
    critic_regularizer_parameter : int = 1 # Num of sampled actions for PBRL, temp for CQL

    # --- experimental OOD filtering in PBRL ---
    filtering_quantile: float = 0.5 # Quantile for filtering in PBRL
    filtering_epsilon: float = 1.5 # Epsilon margin for filtering in PBRL

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

    # --- Additional Logs ---
    diversity_logs: bool = False # Log std and disagreement




"""
    Training state and training step
"""

AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha pretrain_lag train_step")
Transition = namedtuple("Transition", "obs action reward next_obs next_action done")


def create_train_state(args, rng, network, dummy_input, lr=None):
    return TrainState.create(
        apply_fn=network.apply,
        params = network.init(rng, *dummy_input),
        tx=optax.adam(lr if lr is not None else args.lr, eps=1e-5),
        )


def make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset,
                    ood_obs=None, ood_actions=None):
    """
    Make JIT-compatible agent train step.

    """
    print_args(args)

    """
        Get regularizers for ensemble diversity and OOD critic values.
    """
    assert args.pi_operator in ["min", "lcb"]

    ensemble_regularizer_fn = select_regularizer(args, actor_apply_fn, q_apply_fn)
    critic_regularizer_fn = select_ood_regularizer(args, actor_apply_fn, q_apply_fn)

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
            Alpha loss (Entropy temperature)
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
        """
            Update alpha
        """
        updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
        agent_state = agent_state._replace(alpha=updated_alpha)
        alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))
        """
            Actor loss (Policy Improvement)
        """
        @partial(jax.value_and_grad, has_aux=True)
        def _actor_loss_function(params, rng):
            def _compute_loss(rng, transition):
                pi = actor_apply_fn(params, transition.obs)
                sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                log_pi = log_pi.sum()

                q_values = q_apply_fn(
                            agent_state.vec_q.params,
                            transition.obs, sampled_action
                           )
                std_q = q_values.std(-1)

                """
                    Evaluate PI operator
                """
                if args.pi_operator == "min":
                    q_tgt = q_values.min(-1)
                else:
                    # lcb
                    q_tgt = q_values.mean(-1) - args.actor_lcb_penalty * std_q

                return -q_tgt + alpha * log_pi, -log_pi, q_tgt, std_q, sampled_action

            rng = jax.random.split(rng, args.batch_size)
            loss, entropy, q_target, q_std, actions = jax.vmap(_compute_loss)(rng, batch)
            mean_dist = jnp.square(actions - batch.action).mean()

            return loss.mean(), (entropy.mean(), q_target.mean(), q_target.std(), mean_dist) 

        """
            Update actor
        """
        rng, rng_actor = jax.random.split(rng)
        (actor_loss, (entropy, q_mean, q_std, mean_dist)), actor_grad = (
                _actor_loss_function(agent_state.actor.params, rng_actor)
                )

        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        """
            Compute TD targets
        """
        rng_next, rng = jax.random.split(rng, 2)
        next_pis = actor_apply_fn(agent_state.actor.params, batch.next_obs)
        bootstrap_actions, logprobs_next = next_pis.sample_and_log_prob(seed=rng_next)
        logprobs_next = logprobs_next.sum(-1, keepdims=True)

        # --- Bootstrap actions with target nets ---
        next_q = q_apply_fn(agent_state.vec_q_target.params, 
                            batch.next_obs, 
                            bootstrap_actions)
        """
            Evaluate PE operator

            construct bootstrap for each critic Q_i
        """
        if args.shared_targets:
            # [B, E] -> [B, 1] target, later broadcast to [B,E] predictions
            next_v_target = next_q.min(-1, keepdims=True) - alpha * logprobs_next

        else:
            # [B,E] target Q_i'(s',a') - alpha * log pi(a'|s') - beta_id * std_i(Q_J(s',a'))
            std_q_next = next_q.std(-1, keepdims=True)
            next_v_target = next_q - alpha * logprobs_next - args.beta_id * std_q_next


        """
            Construct TD target
                y_ij for every critic Q_i and experience j
        """
        td_target = jnp.expand_dims(batch.reward, -1) + args.gamma * jnp.expand_dims((1 - batch.done), -1) * next_v_target

        # --- Get specialized regularizer loss function with current state --- 
        rng, rng_reg, rng_reg_loss = jax.random.split(rng, 3)
        rng, rng_critic, rng_critic_loss = jax.random.split(rng, 3)

        # --- Construct closures around regularizer functions that sample actions, etc. ---
        ensemble_reg_loss = ensemble_regularizer_fn(agent_state, rng_reg, batch)
        critic_reg_loss = critic_regularizer_fn(agent_state, rng_critic, batch)

        rng, rng_pi = jax.random.split(rng, 2)


        """
            Main critic update (TD+regularizers)
        """
        @partial(jax.value_and_grad, has_aux=True)
        def _q_loss_fn(params):

            q_pred = q_apply_fn(params, batch.obs, batch.action)
            critic_loss = jnp.square((q_pred - td_target))

            """
                TD error

                L(Q_ij) = (Q_ij - y_ij)^2
            """
            critic_loss = critic_loss.sum(-1).mean()

            """
                Ensemble regularizer

                L(Q_ij) += lambda_reg * R(E)
                where E = {Q_1, ..., Q_N} is the ensemble of critics
            """
            regularizer_loss = ensemble_reg_loss(params, rng_reg_loss, batch)

            """
                Critic regularizer 

                L(Q_ij) += R_ood(Q_i, lambda_ood)
            """
            critic_regularizer_loss, logs = critic_reg_loss(q_pred, params, rng_critic_loss, batch)

            critic_loss += args.reg_lagrangian * regularizer_loss
            critic_loss += critic_regularizer_loss

            return critic_loss, (logs,
                                 regularizer_loss,
                                 critic_regularizer_loss, 
                                 q_pred.mean(),
                                 q_pred.std() )

        (critic_loss, (logs, regularizer_loss, critic_regularizer_loss, q_pred_mean,
                       q_pred_std)), critic_grad = _q_loss_fn(agent_state.vec_q.params)


        """
            Update critic ensemble
        """
        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
        agent_state = agent_state._replace(vec_q=updated_q)

        """
            Update target nets (polyak)
        """
        updated_q_target_params = optax.incremental_update(
                agent_state.vec_q.params,
                agent_state.vec_q_target.params,
                args.polyak_step_size,
        )
        updated_q_target = agent_state.vec_q_target.replace(
                params=updated_q_target_params,
        )
        agent_state = agent_state._replace(vec_q_target=updated_q_target)

        # --- Increment step ---
        agent_state = agent_state._replace(train_step=agent_state.train_step + 1)

        loss = {
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "mean_action_dist": mean_dist,
                    "alpha_loss": alpha_loss,
                    "ensemble_regularizer_loss": regularizer_loss,
                    "critic_regularizer_loss": critic_regularizer_loss,
                    "entropy": entropy,
                    "alpha": alpha,
                    "actor_q_mean": q_mean,
                    "actor_q_std": q_std,
                    "q_pred_mean" : q_pred_mean,
                    "q_pred_std": q_pred_std,
                }

        # Add logs from critic regularizer loss
        for k, v in logs.items():
            loss[k] = v


        if args.diversity_logs:

            # --- DIVERSITY: get EDAC diversity loss --- 
            diversity_loss_val = diversity_loss(q_apply_fn, 
                                                agent_state, 
                                                batch.obs, 
                                                batch.action, 
                                                args.num_critics)

            # --- DIVERSITY: sample random actions and eval ---
            rng_perturb, rng = jax.random.split(rng)
            diversity_stats = get_diversity_statistics(q_apply_fn, actor_apply_fn,
                                                       agent_state, rng_perturb,
                                                       batch.obs, batch.action)
            # --- DIVERSITY: get info on OOD data ---
            ood_stats = compute_qvalue_statistics(q_apply_fn,
                                                  agent_state,
                                                  ood_obs, 
                                                  ood_actions)
            # Add to logs
            for k, v in ood_stats.items():
                loss[f"ood_{k}"] = v
            loss["edac_loss"] = diversity_loss_val
            for k, v in diversity_stats.items():
                loss[f"diversity_{k}"] = v
            loss["std_ratio"] = diversity_stats["uniform_q_std"] / diversity_stats["batch_q_std"]

        return (rng, agent_state), loss

    return _train_step



def train(args):

    rng = jax.random.PRNGKey(args.seed)

    # Get timestamped directory
    exp_dir = get_experiment_dirname(args)
    # Save args to JSON inside the main dir
    os.makedirs(exp_dir, exist_ok=True)
    args_path = os.path.join(exp_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(asdict(args), f, indent=2)

    print("Saving experiment data to ", exp_dir)

    """
        Setup OOD dataset if diversity logs are enabled
    """
    if args.diversity_logs:

        """
            get a different dataset, on which we will evaluate 
            ensemble diversity stats (std, disagreement)
        """
        print("Preparing OOD dataset for diversity logs...")
        ood_dataset_name = args.dataset_name.split("-")[0] + "-expert-v2"
        if args.dataset_name == ood_dataset_name:
            # If expert dataset, use medium
            ood_dataset_name = args.dataset_name.split("-")[0] + "-medium-v2"
        rng, rng_ood = jax.random.split(rng)
        ood_obs, ood_actions = prepare_ood_dataset(rng_ood,
                                                  dataset_name=ood_dataset_name,
                                                  ood_samples=50)

        args.log = True # Force logging if diversity logs are enabled
    else:
        # Don't pass any OOD data
        ood_obs, ood_actions = None, None

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
            # --- Rescale rewards ---
            reward=args.reward_scale * jnp.array(dataset["rewards"]) + args.reward_shift,
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
        vec_q=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        vec_q_target=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        alpha=create_train_state(args, rng_alpha, alpha_net, []),
        pretrain_lag=jnp.full((), args.pretrain_lagrangian, dtype=jnp.float32),
        train_step=jnp.full((), 0, dtype=jnp.float32),
    )

    # --- Wrap Q apply fn ---
    q_apply = q_net.apply

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(
        args, actor_net.apply, q_apply, alpha_net.apply, dataset, 
        ood_obs=ood_obs, ood_actions=ood_actions
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

    print("Starting pretraining...")

    if args.pretrain_updates > 0:
        for eval_idx in tqdm.tqdm(range(pretrain_evals), desc="pretrain epochs"):

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

    num_evals = (args.num_updates - args.pretrain_updates) // args.eval_interval

    """
        Offline Training
    """

    print("Starting training")

    for eval_idx in tqdm.tqdm(range(num_evals), desc="train epochs"):
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

        # --- Plot actions --- 
        rng, rng_viz = jax.random.split(rng)

        # Visualizations
        if "bandit" in args.dataset_name:
            visualize_q_vals(args, agent_state, dataset, q_apply, actor_net.apply, rng_viz)
        if "reach" in args.dataset_name:
            visualize_reach_bias(args, agent_state, q_apply, actor_net.apply, rng_viz)



    # Save final checkpoint for evaluation
    if args.checkpoint:
        ckpt_dir = create_checkpoint_dir(exp_dir)
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
        final_returns_dir = f"{exp_dir}/final_returns/"
        os.makedirs(final_returns_dir, exist_ok=True)
        filename = "returns.npz"
        with open(os.path.join(final_returns_dir, filename), "wb") as f:
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
