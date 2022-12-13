from Learner.Learner import *
from step_3.sample_values import *


class UCBLearner(Learner):

    def __init__(self, lamb, secondary, users, n_prices, step=3, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users, n_prices, n_products)

        # upper confidence bounds of arms : media è il conv rate
        # optimistic estimation of the conv rate provided by arms
        self.step = step
        self.means = np.zeros(shape=(self.n_products, self.n_arms))
        self.arm_counters = np.zeros(shape=(self.n_products, self.n_arms))
        self.widths = np.array([[np.inf for _ in range(self.n_arms)] for i in range(self.n_products)])

        # attributes for simulation of expectations on reward
        self.expected_rewards = np.zeros(shape=(self.n_products, self.n_arms))

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users_classes, self.n_arms, step = self.step)

    def act(self):

        if debug:
            print("Expected rewards : \n", self.expected_rewards)

        return np.argmax((self.widths + self.means) * (self.expected_rewards + get_all_margins()), axis=1)

    def update_pulled_and_success(self, price_pulled, product_visited, items_bought, items_rewards):

        estimated_conv_rate = self.estimate_conversion_rates()

        print("ESTIMATED CONV RATE FOR THE NEW USER : \n", estimated_conv_rate)
        # TODO : clipping is wrong ?
        estimated_conv_rate = np.clip(estimated_conv_rate, a_min=0, a_max=1)
        self.users = \
            update_step_parameters_of_simulation(self.users, estimated_conv_rate, product_visited, items_bought,
                                                 self.step)

    def estimate_conversion_rates(self):
        return self.means + self.widths
        #return self.means

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):

        # main update for result presentation
        super().update(price_pulled, reward, None, None, None)

        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)

        sample_conv_rates = compute_sample_conv_rate(product_visited, items_bought)

        # update of confidence bounds

        # update means
        past_averages = self.means[np.arange(0, self.n_products), price_pulled]
        len_averages = self.arm_counters[np.arange(0, self.n_products), price_pulled]
        self.means[np.arange(0, self.n_products), price_pulled] = \
            ((past_averages * len_averages) + sample_conv_rates) / (len_averages + 1)

        # update upper bounds, arm counters of previous day
        for product in range(self.n_products):
            for idx in range(self.n_arms):
                n = len(items_bought[0])
                if self.arm_counters[product, idx] > 0:
                    self.widths[product, idx] = np.sqrt((2 * np.log(self.t)) / self.arm_counters[product, idx])
                else:
                    self.widths[product, idx] = np.inf

        # update counters
        self.arm_counters[np.arange(0, self.n_products), price_pulled] += 1

        for arm_id in range(self.n_arms):
            # TODO : change how price pulled iterate
            simulated_super_arm = np.array([arm_id for _ in range(self.n_products)])
            # simulated_super_arm = price_pulled
            self.sim.prices, self.sim.margins = get_prices_and_margins(simulated_super_arm)
            self.sim.prices_index = simulated_super_arm
            reward, *_ = website_simulation(self.sim, self.users)
            self.expected_rewards[np.arange(self.n_products), simulated_super_arm] = reward

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
