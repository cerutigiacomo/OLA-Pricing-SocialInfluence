from copy import deepcopy
import numpy.random as npr
import numpy as np


class Simulator:
    def __init__(self, prices, margins, lamb, secondary) -> None:
        # self.prices are the price levels set for each product
        self.prices = prices
        # margins matrix associated for each product and each price point
        self.margins = margins
        self.lamb = lamb
        self.visited_primaries = []
        self.secondary_product = secondary

    def simulation(self, j, user_class):
        # This recursive method simulates one user landing on a webpage of one product.
        # Rewards depend on conversion rates, price point, number of items bought, margins and graph_weights
        # secondary products and calls itself recursively to add the rewards of the next primary.

        # The Graph weight has to be set to

        # Compute reward for buying the primary
        rewards = np.zeros(5, np.float16)
        # Conversion rates gives an hypothetical maximum price for that product
        conversion_factor = user_class.conv_rates[j] > npr.random()

        rewards[j] = self.margins[self.prices[j]][j] * \
                     user_class.n_items_bought[self.prices[j]][j] * \
                     conversion_factor  # n_items_bought????

        # Add the current product to the visited ones.
        self.visited_primaries.append(j)

        arr = deepcopy(user_class.graph_weights)[j]   # deepcopy copy all sub-element and not just the pointer
        arr[self.visited_primaries] = 0.0

        if not conversion_factor:
            # Return if the user do not but any item of this product
            return 0

        # FIRST SECONDARY
        first_secondary = int(self.secondary_product[j][0])  # Observation vector for every product!
        if arr[first_secondary] > npr.random():
            # print("going to first secondary",first_secondary,"from prim",j)
            rewards += self.simulation(first_secondary, user_class)

        arr[self.visited_primaries] = 0.0

        # SECOND SECONDARY
        second_secondary = int(self.secondary_product[j][1])
        if (arr[second_secondary] * self.lamb) > npr.random():
            # print("going to second secondary",second_secondary,"from prim",j)
            rewards += self.simulation(second_secondary, user_class)

        # Returns the rewards of that user associated to products bought
        return rewards
