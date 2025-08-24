import gymnasium as gym
import mock_environments
import numpy as np

env = gym.make("SquareReachEnv-v0", H=3, render_mode="human")


# import stable baselines ppo and solve this env
from stable_baselines3 import PPO
# increase entropy penalize
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.2)
model.learn(total_timesteps=100000)



# evaluate the model
obs, info = env.reset()

for i in range(100):
    done = False
    step = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated
        if done:
            obs, info = env.reset()
            print("Episode finished after", step, "steps")
            print("Reward:", reward)

"""
eps = 10
for _ in range(eps):
    obs, info = env.reset()
    done = False
    while not done:
        action = np.array([input("Enter action (-1 to 1): ")], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated
        print("Done:", done, "Reward:", reward)
"""
