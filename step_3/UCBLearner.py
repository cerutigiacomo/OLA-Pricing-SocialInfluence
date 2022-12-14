from Learner.Learner import *
from step_3.sample_values import *
import math


class UCBLearner(Learner):

    def __init__(self, lamb, secondary, users_classes_to_import, n_prices, step=3, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users_classes_to_import, n_prices, n_products)

        # upper confidence bounds of arms : media è il conv rate
        # optimistic estimation of the conv rate provided by arms
        self.step = step
        self.means = np.zeros(shape=(self.n_products, self.n_arms))
        self.arm_counters = np.zeros(shape=(self.n_products, self.n_arms))
        self.widths = np.array([[np.inf for _ in range(self.n_arms)] for i in range(self.n_products)])

        # attributes for simulation of expectations on reward
        self.expected_rewards = np.zeros(shape=(self.n_products, self.n_arms))

        self.bought = np.zeros(shape=(self.n_products, self.n_arms))

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users_classes, self.n_arms, step = self.step)

    def act(self):

        if debug:
            print("Expected rewards : \n", self.expected_rewards)

        #return np.argmax((self.widths + self.means) * (self.expected_rewards + get_all_margins()), axis=1)
        #return np.argmax(self.widths *(get_all_margins()*np.random.uniform(0,self.bought,size=(self.n_products, self.n_arms)) + self.expected_rewards), axis=1)
        return np.argmax(self.expected_rewards, axis=1)

    def update_pulled_and_success(self, price_pulled, product_visited, items_bought, items_rewards):

        estimated_conv_rate = self.estimate_conversion_rates()

        if debug:
            print("ESTIMATED CONV RATE FOR THE NEW USER : \n", estimated_conv_rate)
        # TODO : clipping is wrong ?
        #estimated_conv_rate = np.clip(estimated_conv_rate, a_min=0, a_max=1)
        self.users = \
            update_step_parameters_of_simulation(self.users, estimated_conv_rate, product_visited, items_bought,
                                                 self.step)

    def estimate_conversion_rates(self):
        #return self.means + self.widths
        return self.means

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):

        # main update for result presentation
        super().update(price_pulled, reward, None, None, None)

        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)

        sample_conv_rates = compute_sample_conv_rate(product_visited, items_bought)

        # update of confidence bounds
        self.update_means(price_pulled,sample_conv_rates)
        self.update_bounds()

        self.update_arm_counters(price_pulled)

        # update expectations
        self.update_expected_rewards_by_simulation(price_pulled)
        self.update_boughts(price_pulled, product_visited, items_bought)

        return 0

    def update_means(self, price_pulled, sample_conv_rates):
        # update means
        past_averages = self.means[np.arange(0, self.n_products), price_pulled]
        len_averages = self.arm_counters[np.arange(0, self.n_products), price_pulled]
        self.means[np.arange(0, self.n_products), price_pulled] = \
            ((past_averages * len_averages) + sample_conv_rates) / (len_averages + 1)

    def update_bounds(self):
        for product in range(self.n_products):
            for idx in range(self.n_arms):
                if self.arm_counters[product, idx] > 0:
                    self.widths[product, idx] = np.sqrt((2 * np.log(self.t)) / self.arm_counters[product, idx])
                else:
                    self.widths[product, idx] = np.inf

    def update_arm_counters(self, price_pulled):
        self.arm_counters[np.arange(0, self.n_products), price_pulled] += 1

    def update_expected_rewards_by_simulation(self, price_pulled):
        temp_reward = np.zeros((self.n_products, self.n_arms))
        for product in range(self.n_products):
            for arm_id in range(self.n_arms):
                simulated_super_arm = price_pulled
                simulated_super_arm[product] = arm_id
                self.sim.prices, self.sim.margins = get_prices_and_margins(simulated_super_arm)
                self.sim.prices_index = simulated_super_arm
                reward, *_ = website_simulation(self.sim, self.users)
                temp_reward[np.arange(self.n_products), simulated_super_arm] = reward

        # self.expected_rewards = (self.expected_rewards + temp_reward) /2
        self.expected_rewards = temp_reward

    def update_boughts(self, price_pulled, product_visited, items_bought):
        # TODO : not currently used in the arm selection function -> if used get from user class values ?
        mean_bought = compute_sample_n_bought(product_visited, items_bought) # REFERS TO SAMPLE OF LAST ITERATION
        for product in range(self.n_products):
            if np.isnan(mean_bought)[product]:
                continue
            #self.bought[product, price_pulled[product]] = (self.bought[product, price_pulled[product]] + mean_bought[product])/2 # REFERS TO A TOTAL MEAN OF AL DAYS
            self.bought[product, price_pulled[product]] = mean_bought[product]
        self.bought = np.floor(self.bought)

    def debug(self):
        if debug:
            print("LEARNER BOUNDS ...")
            print("means : \n", self.means)
            print("arms counter : \n", self.arm_counters)
            print("widths : \n", self.widths)
            print("estimated conversion rates : \n", self.estimate_conversion_rates())
