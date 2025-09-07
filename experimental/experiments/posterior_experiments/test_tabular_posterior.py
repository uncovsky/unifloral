# Testing model free posterior


import numpy as np
import gymnasium as gym



# Create a simple environment with 3 states.
# All actions are deterministic
# First satte has two actions that lead to states 1,2
# 1 has two actions, both return normal(1, 0.1) reward and done
# 2 has two actions, both return normal(0, 0.1) reward and done

class SimpleTabularEnv(gym.Env):
    def __init__(self):
        super(SimpleTabularEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(3)
        self.state = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        if self.state == 0:
            if action == 0:
                self.state = 1
            else:
                self.state = 2
            reward = 0
            done = False
        elif self.state == 1:
            reward = np.random.normal(1, 0.1)
            done = True
        elif self.state == 2:
            reward = np.random.normal(0, 0.1)
            done = True
        return self.state, reward, done, False, {}



# Test the environment
N = 1000
prior_mean = 0
prior_cov = 1

# Sample N q-value functions which are normally distributed with mean 0 and
# diagonal covariance - (N, S, A) q values  
q_samples = np.random.normal(prior_mean, prior_cov, (N, 3, 2))

# function for visualizing samples in first state
def plot_samples(samples, title):
    import matplotlib.pyplot as plt
    plt.figure()
    # viz first state
    plt.scatter(samples[:, 0, 0], samples[:, 0, 1], alpha=0.5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('Q-value action 0')
    plt.ylabel('Q-value action 1')
    plt.title(title)
    plt.grid()
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.show()

#plot_samples(q_samples, 'Prior Q-value Samples')


# function to collect dataset from the environment
# unifom and random (s,a,r,s') tuples, return as numpy array of transiiton
# tuples

env = SimpleTabularEnv()

def collect_data(env, num_episodes):
    data = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            data.append((state, int(action), reward, next_state))
            state = next_state
    return data

iterations = [10, 100, 1000]

def shared_bootstrap_targets(transition, q_samples, alpha=0.05, gamma=0.9):

    # Choose based on action
    # s,a,r,s'
    bootstrap = np.min(q_samples[:, transition[3]], axis=0)


def independent_bootstrap_targets(transition, q_samples, alpha=0.05, gamma=0.9):

    bootstraps = q_samples[:, transition[1]] 
    q_samples[:, transition[1]] *= (1 - alpha)
    q_samples[:, transition[1]] += alpha * (transition[2] + gamma * bootstraps)


for size in [1000]:
    data = collect_data(env, size)
    print(data)

    for iteration in iterations:
        init_shared = q_samples.copy()
        init_independent = q_samples.copy()

        plot_samples(init_independent, 'pre indep Q-value Samples')

        for i in range(iteration):
            transition = data[np.random.choice(len(data))]

            shared_bootstrap_targets(transition, init_shared)
            independent_bootstrap_targets(transition, init_independent)

            
        plot_samples(init_independent, 'indep Q-value Samples')
