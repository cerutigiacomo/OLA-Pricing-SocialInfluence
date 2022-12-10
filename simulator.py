from copy import deepcopy
import numpy.random as npr
import numpy as np
from scipy.stats import bernoulli


class Simulator:
    def __init__(self, prices, margins, lamb, secondary, prices_index) -> None:
        # self.prices are the price levels set for each product
        self.prices = prices
        # margins matrix associated for each product and each price point
        self.margins = margins
        self.lamb = lamb
        self.visited_primaries = []
        self.items_bought = np.zeros(5)
        self.items_rewards = np.zeros(5)
        self.secondary_product = secondary
        self.prices_index = prices_index

    def reset(self):
        self.visited_primaries = []
        self.items_bought = np.zeros(5)
        self.items_rewards = np.zeros(5)

    def simulation(self, j, user_class):
        # This recursive method simulates one user landing on a webpage of one product.
        # Rewards depend on conversion rates, price point, number of items bought, margins and graph_weights
        # secondary products and calls itself recursively to add the rewards of the next primary.

        rewards = np.zeros(5, np.float16)

        # bernoullli launch with probability of the user class conversion rate
        conversion_factor = bernoulli.rvs(user_class.conv_rates[j][self.prices_index[j]], size=1)
        items = user_class.get_n_items_to_buy(j)
        rewards[j] = self.margins[j] * \
                     items * \
                     conversion_factor


        # Add the current product to the visited ones.
        self.visited_primaries.append(j)

        arr = deepcopy(user_class.graph_weights)[j]  # deepcopy copy all sub-element and not just the pointer
        arr[self.visited_primaries] = 0.0

        if not conversion_factor:
            # Return if the user do not but any item of this product
            return [0 for _ in range(5)]
        else:
            self.items_bought[j] += items
            self.items_rewards[j] += rewards[j]
        # print("bought: ", user_class.n_items_bought[j], " items of product: ", j)

        # FIRST SECONDARY
        first_secondary = int(self.secondary_product[j][0])  # Observation vector for every product!
        if bernoulli.rvs(arr[first_secondary], size=1):
            # print("going to first secondary",first_secondary,"from prim",j)
            rewards += self.simulation(first_secondary, user_class)

        arr[self.visited_primaries] = 0.0

        # SECOND SECONDARY
        second_secondary = int(self.secondary_product[j][1])
        if bernoulli.rvs(arr[second_secondary]*self.lamb, size=1):
            # print("going to second secondary",second_secondary,"from prim",j)
            rewards += self.simulation(second_secondary, user_class)
        # Returns the rewards of that user associated to products bought
        return rewards
