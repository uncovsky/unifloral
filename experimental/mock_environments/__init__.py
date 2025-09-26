from gymnasium.envs.registration import register
from mock_environments.square_reach import SquareReachEnv
from mock_environments.continuous_bandit import ContinuousBandit

# ==========================
# Register the environments 
# ==========================

register(
    id="SquareReachEnv-v0",
    entry_point="mock_environments.square_reach:SquareReachEnv",
    max_episode_steps=1000,  # default, can be overridden
)


register(
    id="ContinuousBandit-v0",
    entry_point="mock_environments.continuous_bandit:ContinuousBandit",
    max_episode_steps=1,  # one step and then terminal
)



__all__ = ["SquareReachEnv", "ContinuousBandit"]
