import numpy as np

from resources.define_distribution import *
from simulator import Simulator
from website_simulation import *

class Learner:
    def __init__(self, lamb, secondary, users, n_prices, class_indexes, n_products=numbers_of_products,):
        self.lamb = lamb
        self.secondary = secondary
        # data disaggregation is going to be set before this function
        # users would contain a unique Users_group object of aggregated data
        self.users = users

        self.t = 0
        self.prices_index = np.array([0 for i in range(n_products)])
        self.n_products = n_products
        self.n_arms = n_prices
        self.class_indexes = class_indexes

        # TODO REVIEW : DO WE NEED INITIALIZATION OF FOLLOWING DATA ?
        prices, margins = self.get_prices(self.prices_index)
        self.sim = Simulator(prices, margins, lamb, secondary, self.prices_index)
        self.sim.prices, self.sim.margins = self.get_prices(self.prices_index)
        self.reward = website_simulation(self.sim, self.users)

        # list prices used to index x-axis
        self.list_prices = np.append(np.array([]), str(self.prices_index))
        # list margins used to fill y-axis with cumulative rewards
        self.list_margins = np.append(np.array([]), np.sum(self.reward))

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users)

    def act(self):
        # select the next arm to be pulled
        pass

    def update(self, price_pulled, reward):
        # update observation list once reward is returned
        self.t += 1
        self.list_prices = np.append(self.list_prices, str(self.prices_index))
        self.list_margins = np.append(self.list_margins, np.sum(reward))
        self.prices_index = price_pulled

    @staticmethod
    def get_prices(index):
        products = get_product()
        prices = [products[i]["price"][index[i]] for i in range(numbers_of_products)]
        margin = [products[i]["price"][index[i]] - products[i]["cost"] for i in range(numbers_of_products)]
        return prices, margin


class UCBLearner(Learner):

    def __init__(self, lamb, secondary, users, n_prices, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users, n_prices, n_products)

        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.means = np.zeros(shape=(self.n_products, self.n_arms))
        self.arm_counters = np.zeros(shape=(self.n_products, self.n_arms))
        self.widths = np.array([[np.inf for _ in range(self.n_arms)] for i in range(self.n_products)])

    def act(self):
        return np.argmax(self.means + self.widths, axis=1)

    def update(self, price_pulled, reward):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward)

        print("PRICE PULLED : \n", price_pulled)
        print("REWARD OBSERVED : \n", reward)

        # update confidence bounds

        # update means
        past_averages = self.means[np.arange(0,self.n_products), price_pulled]
        len_averages = self.arm_counters[np.arange(0, self.n_products), price_pulled]
        self.means[np.arange(0,self.n_products), price_pulled] = \
            ((past_averages * len_averages) + reward)/(len_averages+1)

        # update counter of selected arms
        self.arm_counters[np.arange(0, self.n_products), price_pulled] += 1

        for product in range(self.n_products):
            for idx in range(self.n_arms):
                if self.arm_counters[product,idx] > 0:
                    self.widths[product,idx] = np.sqrt((2*np.log(self.t))/self.arm_counters[product,idx])
                else:
                    self.widths[product,idx] = np.inf

    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = self.get_prices(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward = website_simulation(self.sim, self.users)
        return observed_reward

    def debug(self):
        print("LEARNER BOUNDS ...")
        print("means : \n",self.means)
        print("arms counter : \n", self.arm_counters)
        print("widths : \n",self.widths)
        print("confidence : \n", self.means+self.widths)