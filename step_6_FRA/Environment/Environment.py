import numpy as np

class Environment():

    def __init__(self, n_arms, probabilities):
        # prbabilities: probability distribution of the arm reward
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        # random binomial reward associated to the pulled arm
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward


