from infra.offline_dataset_wrapper import OfflineDatasetWrapper

from collections import namedtuple
from dataclasses import dataclass, asdict
import d4rl
import distrax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as onp
import optax


# Dummy bc config
@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "bc"
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

# Make a dummy actor
class DeterministicTanhActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        action = nn.Dense(self.num_actions)(x)
        pi = distrax.Transformed(
            distrax.Deterministic(action),
            distrax.Tanh(),
        )
        return pi


def create_train_state(args, rng, network, dummy_input):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(args.lr, eps=1e-5),
    )


AgentTrainState = namedtuple("AgentTrainState", "actor")
Transition = namedtuple("Transition", "obs action reward next_obs done")

def test_d4rl():

    # --- D4RL tests --- 
    args = Args()
    rng = jax.random.PRNGKey(42)
    rng_env_init, rng = jax.random.split(rng)

    wrap = OfflineDatasetWrapper(source="d4rl", dataset=args.dataset)

    # Get dataset and eval env
    dataset = wrap.get_dataset()
    eval_env = wrap.get_eval_env(args.eval_workers, rng_env_init)

    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    print("Dataset:", dataset)


    num_actions = eval_env.single_action_space.shape[0]
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)

    print("Num actions:", num_actions)

    # Create agent state
    rng_actor, rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(eval_env.single_observation_space.shape)

    actor_net = DeterministicTanhActor(num_actions, obs_mean, obs_std)
    agent_state = AgentTrainState(actor=create_train_state(args, rng_actor,
                                                           actor_net,
                                                           [dummy_obs]))

    # Test get_normalized_score
    random_lengths = onp.random.randint(low=1, high=20, size=(20,))
    random_returns = [ onp.random.randn(x) for x in random_lengths ]

    # Test that the normalized score matches D4RL's
    for returns in random_returns:
        score = wrap.get_normalized_score(returns)
        d4rl_score = d4rl.get_normalized_score(args.dataset, returns)
        assert jnp.allclose(score, d4rl_score)


    # Test evaluation
    returns = wrap.eval_agent(args, rng, agent_state)
    scores = wrap.get_normalized_score(returns) * 100.0
    print("Evaluation returns:", returns)
    print("Evaluation scores:", scores)

def test_minari(name):
    # --- Minari tests --- 
    args = Args()
    rng = jax.random.PRNGKey(42)
    rng_env_init, rng = jax.random.split(rng)

    wrap = OfflineDatasetWrapper(source="minari", dataset=name)

    # Get dataset and eval env
    dataset = wrap.get_dataset()

    minari_dataset = wrap.get_minari_dataset()
    assert(minari_dataset is not None)

    eval_env = wrap.get_eval_env(args.eval_workers, rng_env_init)

    dataset = Transition(
        obs=jnp.array(dataset["observations"]),
        action=jnp.array(dataset["actions"]),
        reward=jnp.array(dataset["rewards"]),
        next_obs=jnp.array(dataset["next_observations"]),
        done=jnp.array(dataset["terminals"]),
    )

    print("Dataset:", dataset)


    num_actions = eval_env.single_action_space.shape[0]
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)

    print("Num actions:", num_actions)

    # Create agent state
    rng_actor, rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(eval_env.single_observation_space.shape)

    actor_net = DeterministicTanhActor(num_actions, obs_mean, obs_std)
    agent_state = AgentTrainState(actor=create_train_state(args, rng_actor,
                                                           actor_net,
                                                           [dummy_obs]))

    # Test get_normalized_score
    random_lengths = onp.random.randint(low=1, high=20, size=(20,))
    random_returns = [ onp.random.randn(x) for x in random_lengths ]

    # Test that something sensible is returned
    for returns in random_returns:
        score = wrap.get_normalized_score(returns)
        assert(len(score) == len(returns))

    # Test evaluation
    returns = wrap.eval_agent(args, rng, agent_state)
    scores = wrap.get_normalized_score(returns) * 100.0
    print("Evaluation returns:", returns)
    print("Evaluation scores:", scores)



if __name__ == "__main__":
    names = ["mujoco/halfcheetah/medium-v0", "square-reach/horizon-10-v0"]
    print("Running tests for Minari and D4RL...")
    for name in names:
        print("Testing: ", name)
        test_minari(name)
    print("Minari tests passed.")
    test_d4rl()


