from Learner import Learner
import numpy as np

class TS_Learner(Learner):
    def __init__(self, n_products, n_prices):
        super().__init__(n_products, n_prices)
        self.beta_parameters = np.ones((n_prices, 2))

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, price_pulled, reward, users):
        self.t += 1
        self.update_observations(price_pulled, reward)
        self.beta_parameters[price_pulled, 0] = self.beta_parameters[price_pulled, 0] + reward
        self.beta_parameters[price_pulled, 1] = self.beta_parameters[price_pulled, 1] + 1.0 - reward
