from simulator import *
from website_simulation import *
from resources.define_distribution import *


debug = False

class GreedyReward:
    def __init__(self, lamb, secondary, users) -> None:
        self.lamb = lamb
        self.secondary = secondary
        self.users = users
        self.ite = 0
        self.prices_index = [0, 0, 0, 0, 0]
        # Find the reward with the lowest price
        prices, margins = self.get_prices(self.prices_index)
        self.sim = Simulator(prices, margins, lamb, secondary, self.prices_index)
        self.sim.prices, self.sim.margins = self.get_prices(self.prices_index)
        self.reward = website_simulation(self.sim, self.users)
        if debug:
            print("index:", self.prices_index, "margin: ", self.reward, "sum: ", np.sum(self.reward))
        self.list_prices = np.append(np.array([]), str(self.prices_index))
        self.list_margins = np.append(np.array([]), np.sum(self.reward))

    # There is no guarantee that the algorithm will return the optimal price configuration.
    def bestReward(self):
        self.ite+=1
        rewards = [[] for _ in range(5)]
        # Product to optimize
        j = -1
        max = np.sum(self.reward)
        temp_index = self.prices_index.copy()
        if debug:
            print("try to set one price upper one by one")
        for i in range(5):
            # update the price.
            if temp_index[i] > 2:
                continue

            temp_index[i] += 1
            self.sim.prices, self.sim.margins = self.get_prices(temp_index)
            self.sim.prices_index = temp_index
            # Evaluate the reward for a single arm
            curr_reward = website_simulation(self.sim, self.users)
            rewards[i] = curr_reward
            if debug:
                print("index:", temp_index, "margin: ", curr_reward, "sum: ", np.sum(curr_reward))
            if max < np.sum(curr_reward):
                j = i
                max = np.sum(curr_reward)
            temp_index[i] -= 1

        if debug:
            print("margins: ", [np.sum(rewards[i]) for i in range(5)])
        if np.sum(self.reward) < np.sum(rewards[j]):
            # Choose the best price configuration
            # re-iterate the algorithm.
            self.prices_index[j] += 1
            self.reward = rewards[j]
            self.list_prices = np.append(self.list_prices, str(self.prices_index))
            self.list_margins = np.append(self.list_margins, np.sum(rewards[j]))

            return self.bestReward()

        # If all these price configurations are worse than the configuration in which all the
        # products are priced with the lowest price stop the algorithm and return the configuration
        # with the lowest price for all the products
        return self.reward

    def get_prices(self, index):
        products = get_product()
        prices = [products[i]["price"][index[i]] for i in range(numbers_of_products)]
        margin = [products[i]["price"][index[i]] - products[i]["cost"] for i in range(numbers_of_products)]
        return prices, margin