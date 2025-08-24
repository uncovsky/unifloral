import gymnasium as gym
import mock_environments
import minari

import copy


env = gym.make("SquareReachEnv-v0", H=10)

unwrapped = env.unwrapped



minari_env = minari.load_dataset("square-reach/horizon-10-v0")
eval_env = minari_env.recover_environment(eval_env=True)

# Correct way of makaing the async env
make_eval_env = lambda : minari_env.recover_environment(eval_env=True)
async_env = gym.vector.AsyncVectorEnv([make_eval_env for _ in range(4)])
async_env.reset(seed=[0,1,3,4])
async_env.action_space.seed(0)
print(async_env.step(async_env.action_space.sample()))




