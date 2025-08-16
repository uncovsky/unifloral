import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

class SquareReachEnv(gym.Env):

    """
        Continuous square environment with sparse rewards to test offline
        algorithms and their ability to propagate long horizon rewards.

        The task is a simple, sparse goal-reaching task with controlled horizon
        (step-size). 

        The action space is (-1, 1) which is mappe to angle in pi radians,
        movement is deterministic.

        Horizon is controllable (inversely proportional to the step size).
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, H=10, render_mode=None):
        super().__init__()
        assert H > 0, "Horizon H must be positive."
        self.H = H
        self.step_size = 1.0 / H
        self.render_mode = render_mode

        # Observation: 2D position
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: scalar in (-1, 1) mapped to (-pi, pi)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.goal = np.array([1.0, 1.0], dtype=np.float32)

        self.state = None
        self.t = 0

        # save interaction trajectories
        self.trajectories = []
        self.current_trajectory = []

        self.trajectories_limit = 10

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.t = 0

        if self.current_trajectory and len(self.trajectories) < self.trajectories_limit:
            self.trajectories.append(self.current_trajectory)
        self.current_trajectory = []

        return self.state.copy(), {}

    def step(self, action):

        self.current_trajectory.append(self.state.copy())

        # Clip and map to angle
        action = np.clip(action, -1.0, 1.0)
        theta = float(action[0]) * np.pi

        # Move step_size in chosen direction
        dx = self.step_size * np.cos(theta)
        dy = self.step_size * np.sin(theta)
        self.state = np.clip(self.state + np.array([dx, dy], dtype=np.float32), 0.0, 1.0)

        self.t += 1

        # Check goal condition
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        terminated = dist_to_goal <= self.step_size
        reward = 1.0 if terminated else 0.0

        # truncate on double the effective horizon
        truncated = self.t >= 2 * self.H

        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.t}, Pos: {self.state}")

    def close(self):
        pass

    def plot_trajectories(self):

        plt.figure(figsize=(6, 6))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot(self.goal[0], self.goal[1], 'ro', label='Goal')

        for traj in self.trajectories[:self.trajectories_limit]:
            traj = np.array(traj)

            # plot only dots for each step
            plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=2, alpha=0.5)

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Trajectories in Square Reach Environment')
        plt.legend()
        plt.grid()

        plt.savefig(f"square_horizon={self.H}_trajs.png")



