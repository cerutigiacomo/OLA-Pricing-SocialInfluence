from simulator import Simulator
from website_simulation import *

debug = False


def update_step_parameters_of_simulation(users, estimated_conv_rates, product_visited, items_bought, n_step):
    match n_step:
        case 3:
            # STEP 3
            # unknown : conversion rates
            # known : alpha ratios, number of items bought, graph weights
            users = update_users_conv_rates(users, estimated_conv_rates)

                # STEP 4-5
                # edit n_bought to a fixed maximum ?
                # user.alpha = compute_sample_alpha_ratios()
                # edit alpha ratios
                # edit graph weights

            return users

        case 4:
            # STEP 4
            users = update_users_conv_rates(users, estimated_conv_rates)
            users = update_users_alpha_ratios(users, product_visited)
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

def update_users_alpha_ratios(users, product_visited):
    products_count = np.zeros(numbers_of_products+1)
    products_count[0] = 0.1
    for element in product_visited:
        prod = element[0]
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




class Learner:
    def __init__(self, lamb, secondary, users_classes_to_import, n_prices, n_products=numbers_of_products):
        self.lamb = lamb
        self.secondary = secondary
        # data disaggregation is going to be set before this function
        # Users are reloaded since we could change the properties.

        self.users_classes = users_classes_to_import
        self.users = get_users(users_classes_to_import)

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
        self.__init__(self.lamb, self.secondary, self.users_classes, self.n_arms)


    def act(self):
        # select the next arm to be pulled
        pass

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        # update observation list once reward is returned
        self.t += 1
        self.prices_index = price_pulled
        self.list_prices = np.append(self.list_prices, str(self.prices_index))
        self.list_margins = np.append(self.list_margins, np.sum(reward))

    def update_step_parameters(self, product_visited, items_bought, n_step):
        pass
