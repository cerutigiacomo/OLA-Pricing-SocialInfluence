from simulator import *
from website_simulation import *

class Environment:
    def __init__(self, n_prices, prices, margins, prices_index, users):
        self.n_arms = n_prices
        self.users_indexes = users
        self.users = get_users(users)
        self.prices = prices
        self.sim = Simulator(prices, margins, get_lambda(), get_secondary(), prices_index)

    def round(self, pulled_arm, current_features = None):
        """
        :param current_features: the features of the current considered users
        :param pulled_arm: arm pulled for each product
        :type pulled_arm: list
        :return: reward, product_visited
        :rtype: list
        """
        if current_features is not None:
            self.set_current_features(current_features)
        self.set_margins(pulled_arm)
        reward, product_visited, items_bought, items_rewards = website_simulation(self.sim, self.users)
        return reward, product_visited, items_bought, items_rewards

    def set_margins(self, margin_index):
        self.sim.prices, self.sim.margins = get_prices_and_margins(margin_index)

    def set_current_features(self, current_features):
        self.users = get_users(current_features)

