from simulator import *
from website_simulation import *

class Environment:
    def __init__(self, n_prices, prices, margins, lamb, secondary, prices_index, users_classes, users):
        self.n_arms = n_prices
        self.users_indexes = users_classes
        self.users = users
        self.prices = prices
        self.secondary = secondary
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
        reward_list = []

        reward_tot, product_visited_tot, items_bought_tot, items_rewards_tot = website_simulation(self.sim, self.users)
        reward_list.append(reward_tot)
        for _ in range(35):
            reward, product_visited, items_bought, items_rewards = website_simulation(self.sim, self.users)
            for i,product_visited_by_class in enumerate(product_visited):
                product_visited_tot[i] = product_visited_tot[i] + product_visited_by_class
            for i, items_bought_by_class in enumerate(items_bought):
                items_bought_tot[i] += items_bought_by_class
            for i, items_reward_by_class in enumerate(items_rewards):
                items_rewards_tot[i] += items_reward_by_class
            reward_list.append(reward)
        reward = np.mean(np.array(reward_list), axis=0)
        return reward, product_visited_tot, items_bought_tot, items_rewards_tot

    def set_margins(self, margin_index):
        self.sim.prices, self.sim.margins = get_prices_and_margins(margin_index)

