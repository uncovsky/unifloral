import gymnasium as gym
from gymnasium import spaces
from minari import DataCollector, load_dataset

import mock_environments
import matplotlib.pyplot as plt
import numpy as np

def action_to_goal(obs, gx, gy):
    # Get action to navigate to [gx,gy]
    vx, vy = gx - obs[0], gy - obs[1]
    theta = np.arctan2(vy, vx)
    a = np.array([np.clip(theta / np.pi, -1.0, 1.0)], dtype=np.float32)
    return a


def generate_waypoint():
    # Generate a waypoint in 2/4-th quadrant
    rand = np.random.uniform()
    if rand < 0.5:
        # 2nd quadrant
        gx, gy = np.random.uniform(0.0, 0.5), np.random.uniform(0.5, 1.0)
    else:
        # 4th quadrant
        gx, gy = np.random.uniform(0.5, 1.0), np.random.uniform(0.0, 0.5)

    return gx, gy


def dist(obs, gx, gy):
    return np.sqrt((obs[0] - gx)**2 + (obs[1] - gy)**2)


def collect_dataset(H, random_portion, 
                       episodes=1000,
                       seed=0):

    """
        Collects a dataset on square reach:

        Phase one - sample random waypoints, navigate back to start
                        (confounding trajectories)

        Phase two - sample random waypoint, navigate to goal
    
    """

    assert(0.0 <= random_portion <= 1.0), "random portion must be in [0, 1]"

    env = gym.make("SquareReachEnv-v0", H=H, render_mode="human")

    # collect data
    collecting_env = DataCollector(env)

    """
        Phase one
    """
    # Seed the rng for initial state sampling, but turn it off
    env.unwrapped.set_randomize(False)
    env.unwrapped.seed_init(seed)
    np.random.seed(seed)

    # get step size
    step_size = env.unwrapped.step_size




    random_episodes = int(episodes * random_portion)

    for ep in range(random_episodes):

        obs, _ = collecting_env.reset()
        done = False

        gx, gy = generate_waypoint()
        wp_reached = False


        while not done:
            if not wp_reached:
                # navigate to waypoint deterministically
                a = action_to_goal(obs, gx, gy)

                # will be within step size after this step
                if dist(obs, gx, gy) < 2 * step_size:
                    wp_reached = True
            else:
                a = np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            next_obs, reward, terminated, truncated, _ = collecting_env.step(a)
            obs = next_obs

            done = terminated or truncated


    # phase two

    for ep in range(episodes - random_episodes):

        obs, _ = collecting_env.reset()
        done = False

        gx, gy = generate_waypoint()
        wp_reached = False

        while not done:
            if not wp_reached:
                # navigate to waypoint deterministically
                a = action_to_goal(obs, gx, gy)
                # will be within step size after this step
                if dist(obs, gx, gy) < 2 * step_size:
                    wp_reached = True
            else:
                # navigate to goal 
                a = action_to_goal(obs, 1.0, 1.0)
            next_obs, reward, terminated, truncated, _ = collecting_env.step(a)
            obs = next_obs

            done = terminated or truncated

    env.unwrapped.plot_trajectories()

    name_label = "low"
    if random_portion <= 0.5:
        name_label = "medium"
    if random_portion == 0.0:
        name_label = "expert"

    dataset_id = f"square-reach/horizon-{H}-{name_label}-v0"

    collecting_env.create_dataset(
        dataset_id=dataset_id,
        eval_env=env,
        ref_min_score=0.0,
        ref_max_score=1.0,
        algorithm_name="uniform",
        author="uncovsky",
        description=f"Simple goal reaching env to test reward propagation, horizon {H}, {random_portion} portion of random trajectories")

    env.close()
