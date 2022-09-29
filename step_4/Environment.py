import numpy as np

class Environment():
    def __init__(self, probs, prices):
        self.probs = probs
        self.prices = prices

    def round(self, product, pulled_price):
        conv = np.random.binomial(n=1, p=self.probs[product, pulled_price])
        reward = conv * self.prices[product, pulled_price]
        return reward
