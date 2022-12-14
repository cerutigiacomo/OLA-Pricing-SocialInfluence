import math
from step_6_FRA.Learner import *
from scipy.stats import bernoulli
from resources.define_distribution import *
import json


class UCB_algorithm(Learner):

    def __init__(self, n_arms):
        super().__init__(n_arms)
        # used to count the number of time an arm was selected
        self.numbers_of_selections = [0] * n_arms
        self.empirical_mean = [0] * n_arms

    def pull_arm(self):

        pulled_arm = 0
        max_upper_bound = 0

        for arm in range(0, self.n_arms):
            # if the arm has been already pulled once
            if self.numbers_of_selections[arm] > 0:
                total_counts = sum(self.numbers_of_selections)
                #widths
                bound_length = math.sqrt(0.01 * math.log(total_counts) / float(self.numbers_of_selections[arm]))
                upper_bound = self.empirical_mean[arm] + bound_length
            else:
                upper_bound = 1e400

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                pulled_arm = arm

        return pulled_arm

    def update(self, pulled_arm, reward):

        self.t += 1
        self.update_rewards(pulled_arm, reward)

        self.numbers_of_selections[pulled_arm] = self.numbers_of_selections[pulled_arm] + 1
        n = self.numbers_of_selections[pulled_arm]

        old_empirical_mean = self.empirical_mean[pulled_arm]
        self.empirical_mean[pulled_arm] = ((n - 1) / float(n)) * old_empirical_mean + (1 / float(n)) * reward

    def simulation(self, item, user_class, conv_rates, result):
        # This recursive method simulates one user landing on a webpage of one product.
        # Rewards depend on conversion rates, price point, number of items bought, margins and graph_weights
        # secondary products and calls itself recursively to add the rewards of the next primary.

        f = open('../resources/environment.json')
        data = json.load(f)
        different_value_of_prices = data["product"]["different_value_of_prices"]
        rewards = np.zeros(5, np.float16)
        numbers_of_products = data["product"]["numbers_of_products"]
        today = int(npr.choice(different_value_of_prices))
        prices_index = [today for _ in range(5)]
        products = get_product()

        prices = [products[i]["price"][today] for i in range(numbers_of_products)]
        margins = [products[i]["price"][today] - products[i]["cost"] for i in range(numbers_of_products)]

        # bernoullli launch with probability of the user class conversion rate
        conversion_factor = bernoulli.rvs(conv_rates[item][prices_index[item]], size=1)

        # she/he buys a number of units of the primary product if the price of
        # a single unit is under the user’ reservation price; in other words,
        # the users’ reservation price is not over the cumulative price of
        # the multiple units, but only over the single unit
        if user_class.reservation_price < prices[item]:
            conversion_factor = 0

        items = user_class.get_n_items_to_buy(item)
        rewards[item] = margins[item] * \
                        items * \
                        conversion_factor * \
                        result

        if rewards[item] > 0:
            reward = 1
        else:
            reward = 0


        #print("REWARD per l'item ", j, "é pari a:", rewards[j])

        #max_reward_per_item = []
        #if max_reward_per_item < current_reward
        #    max_reward_per_item = current_reward

        #print("CURRENT:", current_reward, "MAX:", max_reward_per_item)

        # Add the current product to the visited ones.
        #self.visited_primaries.append(item)

        #arr = deepcopy(user_class.graph_weights)[item]  # deepcopy copy all sub-element and not just the pointer
        #arr[self.visited_primaries] = 0.0

        #if not conversion_factor:
            # Return if the user do not but any item of this product
            #return [0 for _ in range(5)]
        #else:
            #self.items_bought[item] += items
            #self.items_rewards[item] += rewards[item]
        #print("bought: ", user_class.n_items_bought[j], " items of product: ", j)

        # FIRST SECONDARY
        #first_secondary = int(self.secondary_product[item][0])  # Observation vector for every product!
        #if bernoulli.rvs(arr[first_secondary], size=1):
            # print("going to first secondary",first_secondary,"from prim",j)
            #rewards += self.simulation(first_secondary, user_class)

        #arr[self.visited_primaries] = 0.0

        # SECOND SECONDARY
        #second_secondary = int(self.secondary_product[item][1])
        #if bernoulli.rvs(arr[second_secondary]*self.lamb, size=1):
            # print("going to second secondary",second_secondary,"from prim",j)
            #rewards += self.simulation(second_secondary, user_class)
        # Returns the rewards of that user associated to products bought

        # Adapt the rewards to the UCB algorithm
        #print ("Array reward", rewards)
        return reward