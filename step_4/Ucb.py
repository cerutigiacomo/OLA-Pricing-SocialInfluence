from Learner import *
import numpy as np

class Ucb(Learner):
    def __init__(self, n_products, n_prices, sim):
        super().__init__(n_products, n_prices)
        self.means = np.zeros(n_products, n_prices)
        self.widths = np.array([np.inf for _ in range(n_prices)] * n_products)
        self.sim = sim

    def act(self):
        idx = np.argmax(self.means + self.widths)
        return idx

    def choose_user(self, users):
        if users != 0:
            product = np.random.randint(0, users.shape())
            if users[product] >= 1:
                return product
            else:
                self.choose_user(users)

    def update(self, price_pulled, reward, users):
        # price_pulled is a list of chosen prices for each product
        # as we have 5 products here it is a list of 5 prices
        j = self.choose_user(users)
        users[j] -= 1
        self.sim.visited_primaries = []
        rewards_j = self.sim.simulation(j, users)
        reward_j = rewards_j > 0
        super().update(j, price_pulled, reward_j)
        self.means = np.mean(self.rewards[j][price_pulled])
        for idx in range(self.n_prices):
            n = len(self.rewards[j][idx])
            if n > 0:
                self.width[j][idx] = np.sqrt(2 * np.log(self.t) / n)
            else:
                self.width[j][idx] = np.inf

