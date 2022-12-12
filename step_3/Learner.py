from simulator import Simulator
from website_simulation import *

debug = True


def update_step_parameters_of_simulation(users, estimated_conv_rates, product_visited, items_bought, n_step):
    match n_step:
        case 3:
            # STEP 3
            # unknown : conversion rates
            # known : alpha ratios, number of items bought, graph weights

            for user in users:
                for prod in range(numbers_of_products):
                    for arm in range(different_value_of_prices):
                        if estimated_conv_rates[prod,arm] != 0.0 :
                            user.conv_rates[prod,arm] = estimated_conv_rates[prod,arm]

                # STEP 4-5
                # edit n_bought to a fixed maximum ?
                # user.alpha = compute_sample_alpha_ratios()
                # edit alpha ratios
                # edit graph weights

            return users

        case 4:
            pass
        case 5:
            pass
        case _:
            raise ValueError()


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
