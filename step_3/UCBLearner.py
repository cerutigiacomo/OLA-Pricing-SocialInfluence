from Learner import *

class UCBLearner(Learner):

    def __init__(self, lamb, secondary, users, n_prices, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users, n_prices, n_products)

        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.means = np.zeros(shape=(self.n_products, self.n_arms))
        self.arm_counters = np.zeros(shape=(self.n_products, self.n_arms))
        self.widths = np.array([[np.inf for _ in range(self.n_arms)] for i in range(self.n_products)])

    def act(self):
        def scale_min_max():
            max_value = np.max(self.means.flatten())
            min_value = np.min(self.means.flatten())
            if self.t > 0 and max_value != min_value :
                x_scaled = (self.means - min_value) / (max_value - min_value)
            else:
                x_scaled = self.means
            return x_scaled
        # TODO : find better solution than simply scaling with current maximum and minimum
        scaled_means = scale_min_max()
        if debug:
            print("SCALED CONFIDENCE ON REWARDS : \n", scaled_means + self.widths)
        # return np.argmax(self.means + self.widths, axis=1)
        return np.argmax(scaled_means + self.widths, axis=1)

    def update(self, price_pulled, reward):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward)
        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)

        # update confidence bounds

        # update means
        past_averages = self.means[np.arange(0, self.n_products), price_pulled]
        len_averages = self.arm_counters[np.arange(0, self.n_products), price_pulled]
        self.means[np.arange(0, self.n_products), price_pulled] = \
            ((past_averages * len_averages) + reward) / (len_averages + 1)

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

    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = get_prices_and_margins(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward, a, b, c = website_simulation(self.sim, self.users)
        return observed_reward

    def debug(self):
        if debug:
            print("LEARNER BOUNDS ...")
            print("means : \n", self.means)
            print("arms counter : \n", self.arm_counters)
            print("widths : \n", self.widths)
            print("confidence : \n", self.means + self.widths)
