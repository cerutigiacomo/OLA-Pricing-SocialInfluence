import numpy as np

from simulator import Simulator
from website_simulation import *

debug = False


def update_step_parameters_of_simulation(users, estimated_conv_rates, product_visited, items_bought, n_step, repeat=False):
    match n_step:
        case 3:
            # STEP 3
            users = update_users_conv_rates(users, estimated_conv_rates)

            return users

        case 4:
            # STEP 4
            users = update_users_conv_rates(users, estimated_conv_rates)
            users = update_users_alpha_ratios(users, product_visited, repeat=repeat)
            users = update_users_max_bought(users, items_bought)
            pass
        case 5:
            # STEP 5
            users = update_users_graph_weights(users, product_visited)
            pass
        case _:
            raise ValueError()

    return users

def update_users_conv_rates(users, estimated_conv_rates):
    for user in users:
        for prod in range(numbers_of_products):
            for arm in range(different_value_of_prices):
                if estimated_conv_rates[prod, arm] != 0:
                    user.conv_rates[prod, arm] = estimated_conv_rates[prod, arm]
    return users

def update_users_alpha_ratios(users, product_visited, repeat=False):
    products_count = np.zeros(numbers_of_products+1)
    products_count[0] = 0.1
    for element in product_visited:
        prod = element[0]
        if repeat:
            for i in range(numbers_of_products):
                products_count[i+1] += element[i]
        else:
            products_count[prod+1] += 1

    products_count[1:] = products_count[1:] / len(product_visited)

    for i in range(numbers_of_products + 1):
        if products_count[i] == 0:
            products_count[i] = 0.001

    for user in users:
        user.alpha = npr.dirichlet(products_count, 1).reshape(numbers_of_products + 1)
    return users

def update_users_max_bought(users, items_bought):
    items_bought_arr = np.array(items_bought)

    # We want to update the max_bought of the users
    # using the items_bought we can make a guess of the max_bought
    # by averaging the items bought != 0
    bought_count = np.zeros(numbers_of_products)
    items_count = np.zeros(numbers_of_products)
    for i in range(len(items_bought_arr)):
        for j in range(numbers_of_products):
            if items_bought_arr[i][j] != 0:
                bought_count[j] += 1
                items_count[j] += items_bought_arr[i][j]
    avg_items_bought = np.zeros(numbers_of_products)

    # If the product was never bought, we set the max_bought to 0
    for i in range(numbers_of_products):
        if bought_count[i] != 0:
            avg_items_bought[i] = items_count[i] / bought_count[i]
        else:
            avg_items_bought[i] = 0

    # we find the maximimum of the average items bought for
    for user in users:
        user.n_items_bought = avg_items_bought
    return users


def update_users_graph_weights(users, product_visited):
    graph = np.zeros((numbers_of_products, numbers_of_products))
    products_count = np.zeros((numbers_of_products, numbers_of_products))
    products_count_view = np.zeros(numbers_of_products)
    for element in product_visited:
        prod = element[0]
        products_count_view[prod] += 1
        for visited in element:
            products_count[prod, visited] += 1

    # Scale the graph weights
    for prod in range(numbers_of_products):
        for sec in range(numbers_of_products):
            products_count[prod][sec] = products_count[prod][sec] / \
                                        (products_count_view[prod] + products_count_view[sec])

    for user in users:
        user.graph = products_count
    return users


def retrieve_items_count(product_visited):
    products_count = np.zeros((numbers_of_products, numbers_of_products))
    products_count_view = np.zeros(numbers_of_products)
    for element in product_visited:
        prod = element[0]
        products_count_view[prod] += 1
        for visited in element:
            products_count[prod, visited] += 1
    return products_count, products_count_view



class Learner:
    def __init__(self, n_prices, n_products=numbers_of_products):

        # data disaggregation is going to be set before this function
        # Users are reloaded since we could change the properties.

        self.pulled_per_arm = np.zeros((n_products, n_prices))
        self.success_by_arm = np.zeros((n_products, n_prices))
        self.t = 1
        self.prices_index = np.array([0 for i in range(n_products)])
        self.n_products = n_products
        self.n_arms = n_prices
        self.current_reward = []

        self.rewards_per_arm = [[[] for i in range(n_prices)] for j in range(n_products)]
        self.boughts_per_arm = [[[0] for i in range(n_prices)] for j in range(n_products)]

        prices, margins = get_prices_and_margins(self.prices_index)
        self.sim = Simulator(prices, margins, get_lambda(), get_secondary(), self.prices_index)
        self.reward = [0 for _ in range(n_products)]

        # list prices used to index x-axis
        self.list_prices = np.array([])
        # list margins used to fill y-axis with cumulative rewards
        self.list_margins = np.array([])
        # list of pulled arm
        self.pulled = []

    def reset(self):
        self.__init__(self.n_arms, self.n_products)


    def act(self):
        # select the next arm to be pulled
        pass

    def update(self, pulled_arm, reward, visited_products, n_bought_products):
        if debug:
            print("pulled_arm: ", pulled_arm,
                  "reward: ", np.sum(reward))

        # update observation list once reward is returned
        self.t += 1
        self.list_prices = np.append(self.list_prices, str(self.prices_index))
        self.list_margins = np.append(self.list_margins, np.sum(reward))


        num_products = np.zeros(self.n_products)
        for i in range(len(n_bought_products[0])):
            for j in range(self.n_products):
                if n_bought_products[0][i][j] != 0:
                    num_products[j] += 1

        if not isinstance(pulled_arm[0], list):
            self.prices_index = pulled_arm
            for prod in range(self.n_products):
                if num_products[prod] == 0:
                    self.rewards_per_arm[prod][pulled_arm[prod]].append(0)
                else:
                    self.rewards_per_arm[prod][pulled_arm[prod]].append(1)
                    self.boughts_per_arm[prod][pulled_arm[prod]].append(num_products[prod])
            self.pulled.append(pulled_arm)
        else:
            if len(pulled_arm[0]) == 0:
                return
            self.prices_index = pulled_arm[0][-1]
            for z in range(len(pulled_arm[0])):
                for prod in range(self.n_products):
                    if num_products[prod] == 0:
                        self.rewards_per_arm[prod][pulled_arm[0][z][prod]].append(0)
                    else:
                        self.rewards_per_arm[prod][pulled_arm[0][z][prod]].append(1)
                        self.boughts_per_arm[prod][pulled_arm[0][z][prod]].append(num_products[prod])
                self.pulled.append(pulled_arm)

    def get_number_of_rewards(self):
        tot = 0
        for i in range(len(self.rewards_per_arm)):
            for j in range(len(self.rewards_per_arm[0])):
                tot += len(self.rewards_per_arm[i][j])
        return tot

    def update_step_parameters(self, product_visited, items_bought, n_step):
        pass
