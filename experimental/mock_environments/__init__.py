from gymnasium.envs.registration import register
from mock_environments.square_reach import SquareReachEnv

# ==========================
# Register the environments 
# ==========================

register(
    id="SquareReachEnv-v0",
    entry_point="mock_environments.square_reach:SquareReachEnv",
    max_episode_steps=1000,  # default, can be overridden
)



__all__ = ["SquareReachEnv"]
