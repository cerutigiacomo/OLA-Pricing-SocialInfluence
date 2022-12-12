from simulator import Simulator
from website_simulation import *


debug = True

class Learner:
    def __init__(self, lamb, secondary, users_classes, n_prices, n_products=numbers_of_products, ):
        self.lamb = lamb
        self.secondary = secondary
        # data disaggregation is going to be set before this function
        # Users are reloaded since we could change the properties.
        self.users = get_users(users_classes)

        self.t = 1
        self.prices_index = np.array([0 for i in range(n_products)])
        self.n_products = n_products
        self.n_arms = n_prices

        # TODO REVIEW : DO WE NEED INITIALIZATION OF FOLLOWING DATA ?
        prices, margins = get_prices_and_margins(self.prices_index)
        self.sim = Simulator(prices, margins, lamb, secondary, self.prices_index)
        self.reward = [0 for _ in range(n_products)]

        # list prices used to index x-axis
        self.list_prices = np.array([])
        # list margins used to fill y-axis with cumulative rewards
        self.list_margins = np.array([])

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users, self.n_arms)

    def act(self):
        # select the next arm to be pulled
        pass

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        # update observation list once reward is returned
        self.t += 1
        self.prices_index = price_pulled
        self.list_prices = np.append(self.list_prices, str(self.prices_index))
        self.list_margins = np.append(self.list_margins, np.sum(reward))
