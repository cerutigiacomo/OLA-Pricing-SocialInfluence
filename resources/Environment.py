from simulator import *
from website_simulation import *

class Environment:
    def __init__(self, n_prices, prices, margins, lamb, secondary, prices_index, class_index):
        self.n_arms = n_prices
        self.users = get_users(class_index)
        self.sim = Simulator(prices, margins, lamb, secondary, prices_index)
        self.lam = lamb

    def round(self, pulled_arm):
        """
        :param pulled_arm: arm pulled for each product
        :type pulled_arm: list
        :return: reward, product_visited
        :rtype: list
        """
        self.set_margins(pulled_arm)
        reward, product_visited, items_bought, items_rewards = website_simulation(self.sim, self.users)
        return reward, product_visited, items_bought, items_rewards

    def set_margins(self, margin_index):
        self.sim.prices, self.sim.margins = get_prices_and_margins(margin_index)

