import numpy as np

class Observer:

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.total_rewards = np.array([])
        self.pulled_arms = np.array([])

    def update_rewards(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.pulled_arms = np.append(self.pulled_arms, float(pulled_arm))
        self.total_rewards = np.append(self.total_rewards, reward)
