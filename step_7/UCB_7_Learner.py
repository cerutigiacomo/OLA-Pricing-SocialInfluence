from Learner import *
from step_3.sample_values import *
import math


class UCBLearner(Learner):

    def __init__(self, users_classes, n_prices, step=3, n_products=numbers_of_products):
        super().__init__(n_prices, n_products)

        self.users_classes = users_classes
        self.users = get_users(users_classes)
        # upper confidence bounds of arms : media è il conv rate
        # optimistic estimation of the conv rate provided by arms
        self.step = step
        self.means = np.zeros(shape=(self.n_products, self.n_arms))
        self.arm_counters = np.zeros(shape=(self.n_products, self.n_arms))
        self.widths = np.array([[np.inf for _ in range(self.n_arms)] for i in range(self.n_products)])

        # attributes for simulation of expectations on reward
        self.expected_rewards = np.zeros(shape=(self.n_products, self.n_arms))



        self.pricesMeanPerProduct = np.mean(get_all_margins(), 1)
        self.means = np.zeros((numbers_of_products, n_prices))
        self.num_product_sold_estimation = np.ones((numbers_of_products, n_prices))
        self.nearbyReward = np.zeros((numbers_of_products, n_prices))
        self.widths = np.ones((numbers_of_products, n_prices)) * np.inf
        self.currentBestArms = np.zeros(n_products)
        self.n = np.zeros((self.n_products, self.n_arms))

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users_classes, self.n_arms, step = self.step)

    def act(self):

        if debug:
            print("Expected rewards : \n", self.expected_rewards)

        return np.argmax((self.widths + self.means) * (self.expected_rewards + get_all_margins()), axis=1)
        # return np.argmax((self.widths + self.means) * ((get_all_margins()*bought) + self.expected_rewards), axis=1)

    def update_pulled_and_success(self, price_pulled, product_visited, items_bought, items_rewards):

        estimated_conv_rate = self.estimate_conversion_rates()

        if debug:
            print("ESTIMATED CONV RATE FOR THE NEW USER : \n", estimated_conv_rate)
        # TODO : clipping is wrong ?
        estimated_conv_rate = np.clip(estimated_conv_rate, a_min=0, a_max=1)
        self.users = \
            update_step_parameters_of_simulation(self.users, estimated_conv_rate, product_visited, items_bought,
                                                 self.step)

    def estimate_conversion_rates(self):
        return self.means + self.widths
        #return self.means

    def update(self, price_pulled, reward, visited_products, num_bought_products):

        # main update for result presentation
        super().update(price_pulled, reward, visited_products, num_bought_products)

        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)

        self.currentBestArms = price_pulled
        for prod in range(self.n_products):
            new_mean = 0
            if len(self.rewards_per_arm[prod][price_pulled[prod]]) > 0:
                new_mean = np.mean(self.rewards_per_arm[prod][price_pulled[prod]])
            if not np.isnan(new_mean):
                self.means[prod][price_pulled[prod]] = new_mean
            sold_estimation = np.mean(self.boughts_per_arm[prod][price_pulled[prod]])
            if not np.isnan(sold_estimation):  # to avoid Nan values in the matrix
                self.num_product_sold_estimation[prod][price_pulled[prod]] = sold_estimation
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                self.n[prod, arm] = len(self.rewards_per_arm[prod][arm])
                if (self.n[prod, arm]) > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(np.log(self.t)) / self.n[prod, arm]))
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = np.zeros((self.n_products, self.n_arms))
        self.num_product_sold_estimation[np.isnan(self.num_product_sold_estimation)] = 1
        self.num_product_sold_estimation[self.num_product_sold_estimation == 0] = 1

        for prod in range(self.n_products):
            for price in range(self.n_arms):
                for temp in range(self.n_products):
                    self.nearbyReward[prod][price] += self.means[prod][price] * \
                                                      self.means[temp][self.currentBestArms[temp]] * \
                                                      self.num_product_sold_estimation[temp][self.currentBestArms[temp]] * \
                                                      get_all_margins()[temp][self.currentBestArms[temp]]
        return 0


    def debug(self):
        if debug:
            print("LEARNER BOUNDS ...")
            print("means : \n", self.means)
            print("arms counter : \n", self.arm_counters)
            print("widths : \n", self.widths)
            print("estimated conversion rates : \n", self.estimate_conversion_rates())


    def update_step(self, a):
        if self.step == 3:
            for i in range(len(users_classes)):
                self.learner.users[i].conv_rates = a
        if self.step == 4:
            for i in range(len(users_classes)):
                self.learner.users[i].alpha = a

        """
        00000 -> 10
        10000 -> 15
        20000 -> 20
        ...
        01000 -> 11
        02000 -> 16
        03000 -> 21
        ...
        00001 -> 12
        00002 -> 17
        00003 -> 22
        
        """

        rew = np.zeros((numbers_of_products, different_value_of_prices))
        for prod in range(numbers_of_products):
            for arm in range(self.n_arms):
                index_prices = np.zeros(numbers_of_products)
                index_prices[prod] = arm
                rew[prod] = self.simulate(index_prices)[prod]


    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = get_prices_and_margins(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward, a, b, c = website_simulation(self.sim, self.users)
        return observed_reward

    def updateHistory(self, reward, arm_pulled, visited_products, num_bought_products, num_primary=None):
        super().update(arm_pulled, reward, visited_products, num_bought_products)

        num_bought_p = np.zeros(numbers_of_products)
        for i in range(len(num_bought_products[0])):
            for j in range(self.n_products):
                num_bought_p[j] += num_bought_products[0][i][j]

        if not isinstance(arm_pulled[0], list):
            current_prices = [i[j] for i, j in zip(get_all_margins(), arm_pulled)]
        else:
            if len(arm_pulled[0]) == 0:
                return
            current_prices = [i[j] for i, j in zip(get_all_margins(), arm_pulled[0][-1])]
        current_reward = sum(num_bought_p * np.array(current_prices))
        self.current_reward.append(current_reward)

    def get_opt_arm_value(self):
        """
        :return: returns the value associated with the optimal arm
        :rtype: float
        """
        aaa = (self.widths + self.means)
        bbb = (get_all_margins() * self.num_product_sold_estimation)
        ccc = aaa * bbb + self.nearbyReward

        return np.max(
            ccc, axis=1)


    def update_for_all_arms(self):
        for prod in range(self.n_products):
            for price in range(self.n_arms):
                if len(self.rewards_per_arm[prod][price]) > 0:
                    new_mean = np.mean(self.rewards_per_arm[prod][price])
                else:
                    new_mean = 0
                self.means[prod][price] = new_mean
                sold_estimation = np.mean(self.boughts_per_arm[prod][price])
                if not np.isnan(sold_estimation):  # to avoid Nan values in the matrix
                    self.num_product_sold_estimation[prod][price] = sold_estimation
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                self.n[prod, arm] = len(self.rewards_per_arm[prod][arm])
                if (self.n[prod, arm]) > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(np.log(self.t)) / self.n[prod, arm]))
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = np.zeros((self.n_products, self.n_arms))
        self.visit_probability_estimation = np.zeros((5, 5))
        self.visit_probability_estimation[np.isnan(self.visit_probability_estimation)] = 0
        self.num_product_sold_estimation[np.isnan(self.num_product_sold_estimation)] = 1
        self.num_product_sold_estimation[self.num_product_sold_estimation == 0] = 1
        for prod in range(self.n_products):
            for price in range(self.n_arms):
                for temp in range(self.n_products):
                    self.nearbyReward[prod][price] = 0

        self.nearbyReward[np.isnan(self.nearbyReward)] = 0



    def isUcb(self):
        return True
