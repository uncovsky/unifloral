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

        # state space is just one state
        self.observation_space = spaces.Box(
            low=1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Action: scalar in (-1, 1)
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

