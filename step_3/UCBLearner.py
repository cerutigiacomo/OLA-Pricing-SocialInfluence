from Learner import *
from step_3.Learner import update_step_parameters_of_simulation
from step_3.sample_values import *


class UCBLearner(Learner):

    def __init__(self, lamb, secondary, users, n_prices, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users, n_prices, n_products)

        # upper confidence bounds of arms : media è il conv rate
        # optimistic estimation of the conv rate provided by arms
        self.means = np.zeros(shape=(self.n_products, self.n_arms))
        self.arm_counters = np.zeros(shape=(self.n_products, self.n_arms))
        self.widths = np.array([[np.inf for _ in range(self.n_arms)] for i in range(self.n_products)])

        # attributes for simulation of expectations on reward
        self.expected_rewards = np.zeros(shape=(self.n_products, self.n_arms))

    def act(self):
        # arms pulled for each product
        # [0 0 0 0 0]
        # [1 1 1 1 1]
        # ...
        # [3 3 3 3 3]

        # reward matrix
        # 00 01 02 03
        # 10 11 12 13
        # ...
        # 40 41 42 43
        # TODO : review solution, improve by adding accuracy ?
        # BUT this is just a sample, i.e. running the simulation for N rounds would return an accurate expectation value

        for arm_id in range(self.n_arms):
            simulated_super_arm = np.array([arm_id for _ in range(self.n_products)])
            # self.sim.prices, self.sim.margins ? TODO ? !!!
            self.sim.prices_index = simulated_super_arm
            reward, *_ = website_simulation(self.sim, self.users)
            self.expected_rewards[np.arange(self.n_products), simulated_super_arm] += reward

        if debug:
            print("Expected rewards : \n", self.expected_rewards)

        new_arm_indexes_to_be_pulled_next_day = np.argmax(self.expected_rewards, axis=1)

        return new_arm_indexes_to_be_pulled_next_day

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        # method for updates of daily observations result

        # main update for result presentation
        super().update(price_pulled, reward, None, None, None)

        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)

        # estimate sample values from daily observation result
        # 1 conversion rate
        sample_conv_rates = compute_sample_conv_rate(product_visited, items_bought)
        # ... TODO: how to manage bounds on other values ? or just use the means for alpha, n_bought, ...

        # update of confidence bounds

        # update means
        past_averages = self.means[np.arange(0, self.n_products), price_pulled]
        len_averages = self.arm_counters[np.arange(0, self.n_products), price_pulled]
        self.means[np.arange(0, self.n_products), price_pulled] = \
            ((past_averages * len_averages) + sample_conv_rates) / (len_averages + 1)

        # update upper bounds, arm counters of previous day
        for product in range(self.n_products):
            for idx in range(self.n_arms):
                if self.arm_counters[product, idx] > 0:
                    self.widths[product, idx] = np.sqrt((2 * np.log(self.t)) / self.arm_counters[product, idx])
                else:
                    self.widths[product, idx] = np.inf

        # update counters
        self.arm_counters[np.arange(0, self.n_products), price_pulled] += 1

    def update_step_parameters(self, product_visited, items_bought, n_step=3):
        estimated_conv_rate = self.estimate_conversion_rates()
        estimated_conv_rate = np.clip(estimated_conv_rate, a_min=0, a_max=1)
        self.users = \
            update_step_parameters_of_simulation(self.users, estimated_conv_rate, product_visited, items_bought, n_step)

    def estimate_conversion_rates(self):
        return self.means + self.widths
        #return np.clip(self.means + self.widths, a_min=0, a_max=1)
        
        
    def update_pulled_and_success(self, pulled_arm, visited, n_bought_products, items_rewards):
        #Need to be implemented ?
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
