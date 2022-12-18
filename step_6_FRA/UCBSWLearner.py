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

    def act(self):
        def scale_min_max(matrix):
            scaled_matrix = np.zeros((self.n_products, self.n_arms))
            for row in range(matrix.shape[0]):
                max = np.max(matrix[row,:])
                min = np.min(matrix[row,:])
                scaled_matrix[row,:] = (matrix[row,:]-min)/(max-min)

            return scaled_matrix

        def scale_mean_std(matrix):
            mean = np.mean(matrix.flatten())
            std = np.std(matrix.flatten())
            return (matrix-mean)/std

        term1 = (0.25) * self.widths
        term2 = (0.60) * (scale_min_max(self.expected_rewards))
        term3 = (0.65) * (scale_min_max(self.last_expected_rewards))

        if self.t == 12 or self.t == 13 or self.t ==14:
            print("ITERAZIONE : ",self.t)
            print("term1 : \n", term1)
            print("tem2 : \n", term2)

        #arm = np.argmax(((1/100)*self.widths) + ((99/100)*scale_min_max(scale_mean_std(self.expected_rewards))), axis=1)
        arm = np.argmax(term1 + term2 + term3, axis=1)
        print("ARM TO BE PULLED : ", arm)
        return arm

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        Learner.update(self, price_pulled, reward, None, None, None)
        print("OBSERVED REWARD : ", reward)
        sample_conv_rates = compute_sample_conv_rate(product_visited, items_bought)
        self.update_means(price_pulled, sample_conv_rates)
        self.update_bounds()
        self.update_arm_counters(price_pulled)
        self.update_expected_rewards_by_simulation(price_pulled)
        self.update_boughts(price_pulled, product_visited, items_bought)

    def update_means(self, price_pulled, sample_conv_rates):
        past_averages = self.means[np.arange(0, self.n_products), price_pulled]

        len_averages = self.len_averages[np.arange(0, self.n_products), price_pulled]
        for product in range(self.n_products):
            if ~np.isnan(sample_conv_rates[product]):
                self.len_averages[product, price_pulled[product]] += 1
        for product in range(self.n_products):
            if ~np.isnan(sample_conv_rates[product]):
                self.means[product, price_pulled[product]] = \
                    ((past_averages[product] * len_averages[product]) + sample_conv_rates[product]) / (
                        self.len_averages[product, price_pulled[product]])

        #arms_counter = np.reshape(self.arm_counters, newshape=(-1, self.n_products, self.n_arms))
        #arms_counter = np.sum(arms_counter, axis=0)

        #len_averages = arms_counter[np.arange(0, self.n_products), price_pulled]
        #self.means[np.arange(0, self.n_products), price_pulled] = \
        #    ((past_averages * len_averages) + sample_conv_rates) / (len_averages + 1)

    def update_bounds(self):
        arms_counter = np.reshape(self.arm_counters, newshape=(-1, self.n_products, self.n_arms))
        if arms_counter.shape[0] >= self.tau:
            sliding_window_counter = arms_counter[-self.tau:, :, :]
        else:
            sliding_window_counter = arms_counter
        arms_counter = np.sum(sliding_window_counter, axis=0)
        arms_counter = np.reshape(arms_counter, newshape=(self.n_products, self.n_arms))

        if self.t == 14: print(arms_counter)
        for product in range(self.n_products):
            for idx in range(self.n_arms):
                if arms_counter[product, idx] > 0:
                    self.widths[product, idx] = np.sqrt((0.1 * np.log(self.t)) / arms_counter[product, idx])
                #else:
                #    self.widths[product, idx] = np.inf

    def update_arm_counters(self, price_pulled):
        arm_counter = np.zeros(shape=(self.n_products, self.n_arms))
        arm_counter[np.arange(self.n_products), price_pulled] += 1
        self.arm_counters = np.append(self.arm_counters, arm_counter)

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users_classes, self.n_arms, self.tau, self.step, self.n_products)

