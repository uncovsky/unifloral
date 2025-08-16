from gymnasium.envs.registration import register

# ==========================
# Register the environments 
# ==========================

register(
    id="SquareReachEnv-v0",
    entry_point="mock_environments.square_reach:SquareReachEnv",
    max_episode_steps=1000,  # default, can be overridden
)

