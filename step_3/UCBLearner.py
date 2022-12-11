from Learner import *


class UCBLearner(Learner):

    def __init__(self, lamb, secondary, users, n_prices, top_reward=None, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users, n_prices, n_products)

        # TODO : testing best reward given the observability of step3's variables
        self.top_reward = top_reward

        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.means = np.zeros(shape=(self.n_products, self.n_arms))
        self.arm_counters = np.zeros(shape=(self.n_products, self.n_arms))
        self.widths = np.array([[np.inf for _ in range(self.n_arms)] for i in range(self.n_products)])

    def act(self):
        # CONV RATES UNKNOWN SOLUTION
        return np.argmax(self.means + self.widths, axis=1)

        def scale_min_max():
            # max_value = np.max(self.means.flatten())
            max_value = self.top_reward
            min_value = np.min(self.means.flatten())
            if self.t > 0 and max_value != min_value:
                x_scaled = (self.means - min_value) / (max_value - min_value)
            else:
                x_scaled = self.means
            return x_scaled

        scaled_means = scale_min_max()
        if debug:
            print("SCALED MEANS : \n", scaled_means)
            print("SCALED CONFIDENCE ON REWARDS : \n", scaled_means + self.widths)
        return np.argmax(scaled_means + self.widths, axis=1)

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward, None, None, None)

        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)

        sample_conv_rates = np.full_like(np.array([]), fill_value=0, shape=self.n_products)
        counters = np.zeros(shape=self.n_products)
        for (i, (visited_i, bought_i, rewards_i)) in enumerate(
                zip(product_visited[0], items_bought[0], items_rewards[0])):
            seen = np.zeros(shape=numbers_of_products)
            seen[visited_i] += 1
            bought = np.zeros(shape=numbers_of_products)
            bought[bought_i > 0.0] += 1
            counters += seen
            mask_seen = seen > 0
            sample_conv_rates[mask_seen] = \
                (sample_conv_rates[mask_seen] * (counters[mask_seen] - 1) + (bought[mask_seen] / seen[mask_seen])) \
                / counters[mask_seen]
            # sample_conv_rates[np.isnan(sample_conv_rates)] = 0

        # update confidence bounds

        # update means
        past_averages = self.means[np.arange(0, self.n_products), price_pulled]
        len_averages = self.arm_counters[np.arange(0, self.n_products), price_pulled]
        # self.means[np.arange(0, self.n_products), price_pulled] = ((past_averages * len_averages) + reward) / (len_averages + 1)
        self.means[np.arange(0, self.n_products), price_pulled] = \
            ((past_averages * len_averages) + sample_conv_rates) / (len_averages + 1)

        # update counter of selected arms
        mask_positive_rewards = np.array([True if x > 0.0 else False for x in reward])
        for product_id in range(numbers_of_products):
            if mask_positive_rewards[product_id]:
                self.arm_counters[product_id, price_pulled[product_id]] += 1
        # update counter of every arm, included the one with observed reward equal 0
        # self.arm_counters[np.arange(0,number_of_products), price_pulled] += 1

        for product in range(self.n_products):
            for idx in range(self.n_arms):
                if self.arm_counters[product, idx] > 0:
                    self.widths[product, idx] = np.sqrt((2 * np.log(self.t)) / self.arm_counters[product, idx])
                else:
                    self.widths[product, idx] = np.inf

    def debug(self):
        if debug:
            print("LEARNER BOUNDS ...")
            print("means : \n", self.means)
            print("arms counter : \n", self.arm_counters)
            print("widths : \n", self.widths)
            print("confidence : \n", self.means + self.widths)
