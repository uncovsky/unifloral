import gymnasium as gym
from minari import DataCollector

from data_collection.pendulum_data_collection import collect_uniform_dataset, collect_agent_dataset, visualize_policy_histogram
from data_collection.pendulum_data_collection import gaussian_mixture_policy, uniform_mixture_policy, uniform_policy
from data_collection.square_maze_data_collection import collect_dataset


if __name__ == "__main__":

    print("Collecting data for Square Maze...")
    horizons = [10, 50, 200]
    dataset_size = 1000

    # ratios of random trajectories in the dataset
    noises = [0.1, 0.2, 0.3]

    seed = 0

    for noise in noises:
        for ratio in ratios:
            print(f"Horizon: {horizon}, gaussian noise: {noise}")
            seed += 1
            collect_dataset(horizon, ratio, dataset_size, seed)

    print("Collecting data for Pendulum-v1...")
    env = gym.make("Pendulum-v1")

    dataset_names = {
        "gaussian_mixture_policy": "gaussian-v0",
        "uniform_mixture_policy": "mixture-v0",
        "uniform_policy": "uniform-v0",
    }

    pendulum_seed = 42


    for policy in [
        gaussian_mixture_policy,
        uniform_mixture_policy,
        uniform_policy,
    ]:
        print(f"Policy: {policy.__name__}")

        visualize_policy_histogram(policy, env, num_samples=10000)

        collect_uniform_dataset(
            env, policy, dataset_size=100000,
            dataset_name=dataset_names[policy.__name__], seed=pendulum_seed
        )

    """
    Train expert agent, need to fix imports, etc.
    sys.argv = ["python", "--algo", "ppo", "--env", "Pendulum-v1", "--n-timesteps", "1000000", "--seed", "42"]
    train()
    path = os.path.abspath('') + '/logs/ppo/Pendulum-v1_1/best_model'
    load model here
    collect_agent(collecting_env, agent, dataset_size=100000,
    dataset_name="ppo-expert-v1", seed=42)
    """


