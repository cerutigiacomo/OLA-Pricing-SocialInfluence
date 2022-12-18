from Learner.Learner import *

reduction_factor = 1 # used to reduce the number of customer during the learner simulation for updating the means

class TSLearner(Learner):

    def __init__(self, lamb, secondary, users_classes, n_prices, n_products=numbers_of_products, step=3):
        super().__init__(lamb, secondary, users_classes, n_prices, n_products)

        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.beta_parameters = np.ones((n_products, n_prices, 2))

        self.step = step
        self.success_by_arm = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm = np.zeros((self.n_products, self.n_arms))
        self.nearby_reward = np.zeros((self.n_products, self.n_arms))
        self.estimed_conv_rate = np.zeros((self.n_products, self.n_arms))
        # load all the margins
        self.margin = get_all_margins()

    def reset(self):
        self.__init__(self.lamb, self.secondary, self.users_classes, self.n_arms, self.n_products, self.step)

    def act(self):
        index = np.array([0 for _ in range(self.n_products)])
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = npr.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            index[prod] = np.argmax(beta * (get_all_margins()[prod] * self.users[0].get_n_items_to_buy(prod) * self.estimed_conv_rate[prod] * self.nearby_reward[prod]))
            if debug:
                np.set_printoptions(precision=2)
                print("prod: ", prod,
                      "beta : ", beta,
                      "nearby_reward : ", self.nearby_reward[prod],
                      "estimed_conv_rate : ", self.estimed_conv_rate[prod])
        return index

    def update(self, price_pulled, reward, product_visited, items_bought, items_rewards):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward, None, None, None)
        if debug:
            print("PRICE PULLED : ", price_pulled)
            print("REWARD OBSERVED : ", np.sum(reward))

        self.beta_parameters[:, :, 0] = self.beta_parameters[:, :, 0] + \
                                        self.success_by_arm[:, :]
        self.beta_parameters[:, :, 1] = self.beta_parameters[:, :, 1] + \
                                        self.pulled_per_arm - self.success_by_arm

        self.pulled_per_arm = np.zeros((self.n_products, self.n_arms))
        self.success_by_arm = np.zeros((self.n_products, self.n_arms))

        # Update the nearby reward by evaluating different prices starting with the price_pulled
        self.nearby_reward_evaluation(price_pulled)


    def nearby_reward_evaluation(self, price_pulled):
        # get the reward for each arm estimated by the simulation

        nearby_reward = np.zeros((self.n_products, self.n_arms))
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                price_index = price_pulled.copy()
                price_index[prod] = arm
                nearby_reward[prod][arm] = self.simulate(price_index)[prod]
        self.nearby_reward = (self.nearby_reward + nearby_reward) / 2

    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = get_prices_and_margins(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward, a, b, c = website_simulation(self.sim, self.users, reduction_factor)
        return observed_reward


    def update_pulled_and_success(self, pulled_arm, visited, n_bought_products, items_rewards):

        # Update success by arm and pulled per arm for beta parameters.
        for i in range(len(visited)):
            for j in range(len(visited[i])):
                prod = visited[i][j]
                if n_bought_products[i][prod] > 0:
                    self.success_by_arm[prod, pulled_arm[prod]] += 1
                self.pulled_per_arm[prod, pulled_arm[prod]] += 1


        counters = np.zeros(shape=self.n_products)

        # If the conv rate == 1, set to 0 to retrieve the mean since probably the arm has not been pulled enough times.
        estimed_conv_rate = self.get_estimed_conv_rate()

        for (i, (visited_i, bought_i)) in enumerate(
                zip(visited, n_bought_products)):
            seen = np.zeros(shape=numbers_of_products)
            seen[visited_i] += 1
            bought = np.zeros(shape=numbers_of_products)
            bought[bought_i > 0.0] += 1
            counters += seen
            mask_seen = seen > 0

            for j in range(self.n_products):
                if mask_seen[j]:
                    a = estimed_conv_rate[j][pulled_arm[j]]
                    if a == 0:
                        # do not consider the old value since it is 0
                        a = 1
                    #if (a * (counters[j] - 1) + (bought[j]) / seen[j]) / (counters[j]) == a and a != 1:
                    #   print("ERROR: new conv_rate is 0\n\n")
                    estimed_conv_rate[j][pulled_arm[j]] = (a * (counters[j] - 1) + (bought[j]) / seen[j]) \
                                                          / (counters[j])

            # update the conv rate, setting 1 to not visited products
            self.set_con_rates_to_1(estimed_conv_rate)

        self.users = \
            update_step_parameters_of_simulation(self.users, estimed_conv_rate, visited, n_bought_products,
                                                 self.step)

    def get_estimed_conv_rate(self):
        estimed_conv_rate = np.ones((self.n_products, self.n_arms))
        for i in range(self.n_products):
            for j in range(self.n_arms):
                if self.estimed_conv_rate[i][j] == 1:
                    estimed_conv_rate[i][j] = 0
                else:
                    estimed_conv_rate[i][j] = self.estimed_conv_rate[i][j]
        return estimed_conv_rate

    def set_con_rates_to_1(self, conv_rates):
        for i in range(self.n_products):
            for j in range(self.n_arms):
                if conv_rates[i][j] == 0:
                    self.estimed_conv_rate[i][j] = 1
                else:
                    self.estimed_conv_rate[i][j] = conv_rates[i][j]

    def debug(self):
        pass
