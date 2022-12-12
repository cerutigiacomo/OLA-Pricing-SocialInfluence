from Learner import *

class TSLearner(Learner):

    def __init__(self, lamb, secondary, users_classes, n_prices, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users_classes, n_prices, n_products)

        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.beta_parameters = np.ones((n_products, n_prices, 2))

        self.success_by_arm = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm = np.zeros((self.n_products, self.n_arms))
        self.nearby_reward = np.zeros((self.n_products, self.n_arms))
        # load all the margins
        self.margin = get_all_margins()

    def act(self):
        index = np.array([0 for _ in range(self.n_products)])
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = npr.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            index[prod] = np.argmax(beta * (self.margin[prod] + self.nearby_reward[prod]))
        return index

    def update_pulled_and_success(self, pulled_arm, visited, n_bought_products, items_rewards):
        for i in range(len(visited[0])):
            for j in range(len(visited[0][i])):
                prod = visited[0][i][j]
                if n_bought_products[0][i][prod] > 0:
                    self.success_by_arm[prod, pulled_arm[prod]] += 1
                self.pulled_per_arm[prod, pulled_arm[prod]] += 1

        # Estimate the sample_conversion rate for each product
        self.sim.sample_conv_rates = np.full_like(np.array([]), fill_value=0, shape=self.n_products)
        counters = np.zeros(shape=self.n_products)
        for (i, (visited_i, bought_i, rewards_i)) in enumerate(
                zip(visited[0], n_bought_products[0], items_rewards[0])):
            seen = np.zeros(shape=numbers_of_products)
            seen[visited_i] += 1
            bought = np.zeros(shape=numbers_of_products)
            bought[bought_i > 0.0] += 1
            counters += seen
            mask_seen = seen > 0
            self.sim.sample_conv_rates[mask_seen] = \
                (self.sim.sample_conv_rates[mask_seen] * (counters[mask_seen] - 1) + (
                            bought[mask_seen] / seen[mask_seen])) \
                / counters[mask_seen]
            # sample_conv_rates[np.isnan(sample_conv_rates)] = 0

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward, None, None, None)
        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)


        # TODO update beta parameters
        self.beta_parameters[:, :, 0] = self.beta_parameters[:, :, 0] + \
                                        self.success_by_arm[:, :]
        self.beta_parameters[:, :, 1] = self.beta_parameters[:, :, 1] + \
                                        self.pulled_per_arm - self.success_by_arm

        self.pulled_per_arm = np.zeros((self.n_products, self.n_arms))
        self.success_by_arm = np.zeros((self.n_products, self.n_arms))

        # Update the nearby reward by evaluating different prices starting with the price_pulled
        self.nearby_reward_evaluation(price_pulled)


    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = get_prices_and_margins(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward, a, b, c = website_simulation(self.sim, self.users)
        return observed_reward

    def debug(self):
        if debug:
            print("beta par : \n", self.beta_parameters)


    def nearby_reward_evaluation(self, price_pulled):
        # get the reward for each arm estimated by the simulation
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                price_index = price_pulled
                price_index[prod] = arm
                self.nearby_reward[prod][arm] = self.simulate(price_index)[prod]

        if debug:
            print("nearby_reward: ", self.nearby_reward)

