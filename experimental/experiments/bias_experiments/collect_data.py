from data_collection.square_maze_data_collection import collect_dataset
from data_collection.bandit_data_collection import collect_bandit_data

if __name__ == "__main__":

    print("Collecting data for Square Maze...")
    horizons = [10, 50, 200]
    dataset_size = 1000

    # ratios of random trajectories in the dataset
    noises = [0.1, 0.2, 0.3]
    seed = 0

    for horizon in horizons:
        for noise in noises:
            print(f"Horizon: {horizon}, gaussian noise: {noise}")
            seed += 1
            collect_dataset(horizon, noise, dataset_size, seed)


    print("Collecting data for Bandit...")
    collect_bandit_data(episodes=1000, seed=0)
