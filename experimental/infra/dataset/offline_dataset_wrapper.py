import d4rl
import jax
import jax.numpy as jnp
import gymnasium as gymn
import gym
import numpy as onp
import minari
import tqdm
import warnings


class OfflineDatasetWrapper:
    """
        What is needed:

            __init__(source \in "d4rl, minari [ogbench?]", dataset_name)

            load dataset based on src/name

            minari to qlearning dataset fn

            qlearning_dataset (from d4rl)
            
            preprocessing of the dataset:
                add sarsa next actoins
                add return-to-go
                    - do outside of this class.? 

            normalized rewards

            evaluation of agent (return function that can be called with
            policy), given number of workers (async)
    """


    def __init__(self, source : str, dataset : str) -> None:

        # source \in ["d4rl", "minari"]
        self.source = source.lower()
        self.dataset_name = dataset

        # one check for dataset source validity
        if self.source not in ["d4rl", "minari"]:
            raise ValueError(f"Unsupported source of dataset: {source.lower()}")

        # If using minari, store also the original Minari dataset
        self._minari_dataset = None

        # Dataset stored in default D4RL format
        self.dataset = self.setup_dataset(self.source, self.dataset_name)

        # Eval env is set on the first call to evaluate_agent
        self.eval_env = None

        self._fallback_warned = False

    """
        Dataset setup
    """

    def setup_dataset(self, source : str, name : str) -> dict[str, onp.ndarray]:
        """
            Does all necessary conversion and  
            Returns dict with keys:
                observations, actions, rewards, next_observations, terminals

                i.e. the d4rl.qlearning format
        """
        if source == "d4rl":
            return d4rl.qlearning_dataset(gym.make(name))
        # else minari
        return self._setup_minari_dataset(name)

    def _setup_minari_dataset(self, name : str) -> dict[str, onp.ndarray]:
        """
            Loads minari dataset
            Computes min/max theoretical returns
            Converts data into d4rl format, see return
            Returns dict with keys:
                observations, actions, rewards, next_observations, terminals

        WARN: Loads the whole dataset into memory!
            This is not a problem for small datasets, but could be an issue later.
        """

        minari_dataset = minari.load_dataset(name, download=True)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []

        min_length, max_length = float("inf"), float("-inf")
        min_reward, max_reward = float("inf"), float("-inf")

        for episode in tqdm.tqdm(minari_dataset.iterate_episodes(), desc="Loading minari dataset"):
            observations.extend(episode.observations)
            actions.extend(episode.actions)
            rewards.extend(episode.rewards)
            next_observations.extend(episode.observations[1:])
            terminals.extend(episode.terminations)

            # calculate statistics
            min_length = min(min_length, len(episode.observations))
            max_length = max(max_length, len(episode.observations))
            min_reward = min(min_reward, onp.min(episode.rewards))
            max_reward = max(max_reward, onp.max(episode.rewards))


        """
            While all d4rl environments have a normalized score, this is
            optional for minari datasets. Thus we compute a fallback score
            here:

            max_reward in dataset * max length of episode
            min_reward in dataset * min length of episode
        """

        # store the minari dataset for later use
        self._minari_dataset = minari_dataset
        self._fallback_max_reward = max_reward * max_length
        self._fallback_min_reward = min_reward * min_length

        dataset = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminals": terminals,
        }

        return dataset

    """
        Eval / Score methods
    """
    def _initialize_eval_env(self, num_workers, rng):
        rng_actions, rng = jax.random.split(rng)

        # Initialize the evaluation env
        if self.source == "d4rl":
            # Create a new environment for evaluation
            self.eval_env = gym.vector.make(self.dataset_name, num_envs=num_workers)
            return 

        # Else minari
        else:
            self.eval_env = gymn.vector.AsyncVectorEnv([
                lambda: self._minari_dataset.recover_environment(eval_env=True) \
                        for _ in range(num_workers)
            ])

        self.eval_env.action_space.seed(rng_to_integer_seed(rng_actions))


    
    def get_normalized_score(self, returns : onp.ndarray):
        """
            Returns a number between 0 and 1, quantifying performance of the 
            agent on the offline dataset.

            If no min/max reward is provided (minari), then the one computed
            from dataset, see _setup_minari_dataset, is used.
        """
        if self.source == "d4rl":
            return d4rl.get_normalized_score(self.dataset_name, returns)

        # else minari
        try:
            return minari.get_normalized_score(self._minari_dataset, returns)

        except ValueError:
            # use fallback normalization
            if not self._fallback_warned:
                warnings.warn(
                    "No min/max reward provided for minari dataset, using fallback normalization."
                )
                self._fallback_warned = True
            return (returns - self._fallback_min_reward) / (
                self._fallback_max_reward - self._fallback_min_reward

            )

    def eval_agent(self, args, rng, agent_state):
        """
            Evaluates the agent on eval_env for the dataset. For d4rl this env
            is newly created, for minari we recover eval_env from minari
            dataset.

            First call to this method creates the evaluation env
            (AsyncVectorEnv with args.eval_workers workers) and seeds its
            action_space.

            One episode is ran for each worker, we return list of cum. returns
        """

        # Gets eval env, potentially initializes it
        eval_env = self.get_eval_env(args, rng)        

        # Evaluate the agent
        if self.source == "d4rl":
            return eval_agent_gym(args, rng, eval_env, agent_state)

        # Else minari
        return eval_agent_gymnasium(args, rng, eval_env, agent_state)
        


    """
        Getters
    """

    def get_dataset(self) -> dict[str, onp.ndarray]:
        """
            Returns the dataset in d4rl.qlearning format
        """
        return self.dataset


    def get_minari_dataset(self):
        """
            Returns the original minari dataset
            If not using minari, throws exception
        """
        if self.source == "minari":
            return self._minari_dataset

        raise ValueError(
            "This dataset was not loaded from minari, thus no minari dataset is available."
        )


    def get_eval_env(self, args, rng):
        """
            Returns the evaluation environment.
            If not initialized, initializes it first.
        """
        if self.eval_env is None:
            self._initialize_eval_env(args, rng)

        return self.eval_env
        

""" 
    Utils and eval functions for gym/gymnasium environments.
"""


# Transform JAX rng key to integer seed for gym envs
def rng_to_integer_seed(rng):
    return int(jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max))


def eval_agent_gym(args, rng, env, agent_state):
    """ 
        Evaluation function that is consistent with gym (D4RL) old API
        Assuming env is gym.vector.AsyncVectorEnv or gym.vector.SyncVectorEnv
        created with args.eval_workers workers.
    """
    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)

    # Get a list of integer seeds from jax rng
    seeds_reset = [rng_to_integer_seed(rng) for rng in rng_reset]

    env_name = env.env_fns[0]().spec.name
    env_lower = env_name.lower()
    mujoco_envs = ["halfcheetah", "hopper", "walker2d"]
    is_mujoco = any([name in env_lower for name in mujoco_envs])

    if not is_mujoco:
        warnings.warn("Seeding not supported for non-mujooc envs, eval is nondeterministic")
        obs = env.reset()
    else:
        obs = env.reset(seed=seeds_reset)
    # --- Rollout agent ---
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs, eval=True)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action).clip(-1.0, 1.0)

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


def eval_agent_gymnasium(args, rng, env, agent_state):
    """ 
        Evaluation function that is consistent with new gymnasium (Minari) API.
        Assuming env is gymnasium.vector.AsyncVectorEnv or gymnasium.vector.SyncVectorEnv
        created with args.eval_workers workers.
    """
    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)

    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)

    def _rng_to_integer_seed(rng):
        return int(jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max))

    seeds_reset = [rng_to_integer_seed(rng) for rng in rng_reset]

    obs, _ = env.reset(seed=seeds_reset)

    # --- Rollout agent ---
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs, eval=True)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action).clip(-1.0, 1.0)

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        # --- Take step in environment ---
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, terminated, truncated, info = env.step(onp.array(action))


        # --- Update cumulative reward ---
        cum_reward += reward * ~returned

        # --- Track cumulative reward ---
        returned |= terminated | truncated
    return cum_reward

