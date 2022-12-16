import numpy as np
from Learner.Learner import Learner
from resources.define_distribution import numbers_of_products
from step_3.UCBLearner import UCBLearner
from step_3.sample_values import compute_sample_conv_rate


class UCBSWLearner(UCBLearner):
    def __init__(self, lamb, secondary, user_classes_to_import, n_prices, tau, step=3, n_products=numbers_of_products):
        super().__init__(lamb, secondary, user_classes_to_import, n_prices, step, n_products)
        self.arm_counters = np.zeros(shape=(1, self.n_products, self.n_arms))
        self.tau = tau

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        Learner.update(self, price_pulled, reward, None, None, None)
        sample_conv_rates = compute_sample_conv_rate(product_visited, items_bought)
        self.update_means(price_pulled, sample_conv_rates)
        self.update_bounds()
        self.update_arm_counters(price_pulled)
        self.update_expected_rewards_by_simulation(price_pulled)
        self.update_boughts(price_pulled, product_visited, items_bought)

    def update_means(self, price_pulled, sample_conv_rates):
        past_averages = self.means[np.arange(0, self.n_products), price_pulled]

        arms_counter = np.reshape(self.arm_counters, newshape=(-1, self.n_products, self.n_arms))
        arms_counter = np.sum(arms_counter, axis=0)

        len_averages = arms_counter[np.arange(0, self.n_products), price_pulled]
        self.means[np.arange(0, self.n_products), price_pulled] = \
            ((past_averages * len_averages) + sample_conv_rates) / (len_averages + 1)

    def update_bounds(self):
        arms_counter = np.reshape(self.arm_counters, newshape=(-1, self.n_products, self.n_arms))
        if arms_counter.shape[0] >= self.tau:
            sliding_window_counter = arms_counter[-self.tau:, :, :]
        else:
            sliding_window_counter = arms_counter
        arms_counter = np.sum(sliding_window_counter, axis=0)
        arms_counter = np.reshape(arms_counter, newshape=(self.n_products, self.n_arms))

        for product in range(self.n_products):
            for idx in range(self.n_arms):
                if arms_counter[product, idx] > 0:
                    self.widths[product, idx] = np.sqrt((2 * np.log(self.t)) / arms_counter[product, idx])
                else:
                    self.widths[product, idx] = np.inf

    def update_arm_counters(self, price_pulled):
        arm_counter = np.zeros(shape=(self.n_products, self.n_arms))
        arm_counter[np.arange(self.n_products), price_pulled] += 1
        self.arm_counters = np.append(self.arm_counters, arm_counter)

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users_classes, self.n_arms, self.tau, self.step, self.n_products)

