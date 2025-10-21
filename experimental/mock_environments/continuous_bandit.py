import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np
import os

class ContinuousBandit(gym.Env):
    """
        1D continuous bandit, state space is 0 (one state and then terminal)
        Action space is continuous in (-1, 1)

        Rewards are constant 1 for actions in (-1, -0.5) and (0.5, 1)
        And -1 otherwise.
    """
    

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.state = np.array([1.0], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} out of bounds"
        scalar_action = float(action[0])  # convert safely
        if -1.0 <= scalar_action < -0.5 or 0.5 < scalar_action <= 1.0:
            reward = 1.0
        else:
            reward = -1.0
        return self.state, reward, True, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.state, {}


class DDimensionalBandit(gym.Env):
    """
        Generalization to d-dimensional continuous bandit.

        actions in d-dimensional l_inf ball (default eps=0.1) around origin
        have reward +1, otherwise -1.
    """

    def __init__(self, d: int, epsilon: float = 0.1, render_mode=None):

        assert d >= 1 and isinstance(d, int)
        assert 0.0 < epsilon <= np.sqrt(d)  
        super().__init__()

        self.d = d
        self.epsilon = float(epsilon)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.full((d,), -1.0, dtype=np.float32),
            high=np.full((d,),  1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.state = np.array([1.0], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape == ():  # scalar case
            action = action.reshape((1,))
        action = np.clip(action, self.action_space.low, self.action_space.high)

        assert self.action_space.contains(action), f"Action {action} out of bounds for shape {self.action_space.shape}"

        norm = float(np.max(np.abs(action)))
        reward = 1.0 if norm <= self.epsilon else -1.0

        terminated = True
        truncated = False
        info = {"action_norm": norm}
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.state, {}
