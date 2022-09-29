from simulator import *


class GreedyReward:
    def __init__(self, prices, margins, lamb, users) -> None:
        self.prices = prices
        self.margins = margins
        self.lamb = lamb
        self.users = users
        self.ite = 0
        # Find the reward with the lowest price
        self.sim = Simulator(prices, margins, lamb)
        self.reward = self.website_simulation(self.sim, self.users)
        self.list_prices = np.array([])
        self.list_margins = np.array([])
        self.list_prices = np.append(self.list_prices, str(prices))
        self.list_margins = np.append(self.list_margins, np.sum(self.reward))

    # There is no guarantee that the algorithm will return the optimal price configuration.
    def bestReward(self):
        self.ite+=1
        rewards = [[] for _ in range(5)]
        j = -1
        max = np.sum(self.reward)
        for i in range(5):
            # update the price.
            if self.prices[i] > 2:
                continue
            prices = self.prices.copy()
            prices[i] += 1
            self.sim = Simulator(prices, self.margins, self.lamb)
            # Evaluate the reward for a single arm
            curr_reward = self.website_simulation(self.sim, self.users)
            rewards[i] = curr_reward
            if max < np.sum(curr_reward):
                j = i
                max = np.sum(curr_reward)

        if np.sum(self.reward) < np.sum(rewards[j]):
            # Choose the best price configuration and
            # re-iterate the algorithm.
            self.prices[j] += 1
            self.reward = rewards[j]
            self.list_prices = np.append(self.list_prices, str(prices))
            self.list_margins = np.append(self.list_margins, np.sum(rewards[j]))
            #print("Choosing this arm ", self.prices, np.sum(rewards[j]))
            return self.bestReward()

        # If all these price configurations are worse than the configuration in which all the
        # products are priced with the lowest price stop the algorithm and return the configuration
        # with the lowest price for all the products
        return self.reward

    def website_simulation(self, sim, user_class):
        # This method simulates users visiting the ecommerce website
        # argument is an User class instance
        # returns total rewards for all five products

        total_rewards = np.zeros(5, np.float16)

        for j in range(len(user_class.n_users) - 1):
            # Reward of the single product
            product_reward = np.zeros(5, np.float16)
            for n in range(round(user_class.n_users[j + 1])):
                sim.visited_primaries = []
                rewards = sim.simulation(j, user_class)
                product_reward += rewards

            total_rewards += product_reward
            # print(round(user_class.n_users[j+1]),"users landing on product", j+1 ,product_reward)

        return total_rewards
