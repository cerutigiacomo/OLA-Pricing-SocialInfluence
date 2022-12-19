from step_7.Learner import *

reduction_factor = 1 # used to reduce the number of customer during the learner simulation for updating the means

class TSLearner(Learner):

    def __init__(self, users_classes, n_prices, interaction, step=4, n_products=numbers_of_products):
        super().__init__(n_prices, n_products)

        self.average_reward = []
        self.step = step
        self.users_classes = users_classes
        self.users = get_users(users_classes)
        self.interaction = interaction
        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.beta_parameters = np.ones((n_products, n_prices, 2))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))

        self.nearby_reward = np.zeros((self.n_products, self.n_arms))
        self.estimed_conv_rate = np.zeros((self.n_products, self.n_arms))
        # load all the margins
        self.margin = get_all_margins()

    def reset(self):
        self.__init__(self.users_classes, self.n_arms, self.n_products, step=self.step)

    def act(self):
        index = np.array([0 for _ in range(self.n_products)])
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = npr.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            index[prod] = np.argmax(beta *
                                    (get_all_margins()[prod] * self.users[0].get_n_items_to_buy(prod) *
                                     self.estimed_conv_rate[prod] * self.nearby_reward[prod]))
        if not debug:
            np.set_printoptions(precision=2)
            print("index: ", index,
                  "\nnearby_reward : \n", self.nearby_reward,
                  "\nestimed_conv_rate : \n", self.estimed_conv_rate)
        return index

    def update_pulled_and_success(self, pulled_arm, visited, n_bought_products, items_rewards, repeat=False):

        counters = np.zeros(shape=self.n_products)

        # If the conv rate == 1, set to 0 to retrieve the mean since probably the arm has not been pulled enough times.
        estimed_conv_rate = self.get_estimed_conv_rate()

        for (i, (visited_i, bought_i)) in enumerate(
                zip(visited[0], n_bought_products[0])):
            seen = np.zeros(shape=numbers_of_products)
            seen[np.array(visited_i).astype(int)] += 1
            bought = np.zeros(shape=numbers_of_products)
            bought[np.array(bought_i).astype(int) > 0.0] += 1
            counters += seen
            mask_seen = seen > 0

            for j in range(self.n_products):
                if mask_seen[j]:
                    if repeat:
                        for z in range(len(pulled_arm[0])):
                            a = estimed_conv_rate[j][pulled_arm[0][z][j]]
                            if a == 0:
                                # do not consider the old value since it is 0
                                a = 1
                            # if (a * (counters[j] - 1) + (bought[j]) / seen[j]) / (counters[j]) == a and a != 1:
                            #   print("ERROR: new conv_rate is 0\n\n")
                            estimed_conv_rate[j][pulled_arm[0][z][j]] = (a * (counters[j] - 1) + (bought[j]) / seen[j]) \
                                                                  / (counters[j])
                    else:
                        a = estimed_conv_rate[j][pulled_arm[j]]
                        if a == 0:
                            # do not consider the old value since it is 0
                            a = 1
                        # if (a * (counters[j] - 1) + (bought[j]) / seen[j]) / (counters[j]) == a and a != 1:
                        #   print("ERROR: new conv_rate is 0\n\n")
                        estimed_conv_rate[j][pulled_arm[j]] = (a * (counters[j] - 1) + (bought[j]) / seen[j]) \
                                                                 / (counters[j])

            # update the conv rate, setting 1 to not visited products
            self.set_con_rates_to_1(estimed_conv_rate)

        self.users = \
            update_step_parameters_of_simulation(self.users, estimed_conv_rate, visited[0], n_bought_products[0],
                                                 self.step, repeat=repeat)

    def debug(self):
        pass


    def updateHistory(self, reward, arm_pulled, visited_products, num_bought_products, num_primary=None, repeat=None):
        super().update(arm_pulled, reward, visited_products, num_bought_products)

        num_bought_p = np.zeros(numbers_of_products)
        for i in range(len(num_bought_products[0])):
            for j in range(self.n_products):
                num_bought_p[j] += num_bought_products[0][i][j]

        if not isinstance(arm_pulled[0], list):
            #current_prices = [i[j] for i, j in zip(get_all_margins(), arm_pulled)]
            current_reward = np.sum(reward.copy()) #sum(num_bought_p * np.array(current_prices))
        else:
            if len(arm_pulled[0]) == 0:
                return
            #current_prices = [i[j] for i, j in zip(get_all_margins(), arm_pulled[0][-1])]
            current_reward = np.sum(reward.copy())

        self.current_reward.append(current_reward)

        for iter in range(len(num_bought_products[0])):
            if iter >= len(num_bought_products[0]):
                return
            for p in range(numbers_of_products):
                if p >= numbers_of_products:
                    break
                if np.all([len(visited_products[0][z]) == 5 for z in range(len(visited_products[0]))]):
                    if visited_products[0][iter][p] > 0:
                        if num_bought_products[0][iter][p] > 1:
                            self.success_per_arm_batch[p, arm_pulled[0][iter][p]] += 1
                        self.pulled_per_arm_batch[p, arm_pulled[0][iter][p]] += 1
                else:
                    if p in visited_products[0][iter]:
                        if num_bought_products[0][iter][p] > 1:
                            self.success_per_arm_batch[p, arm_pulled[p]] += 1
                        self.pulled_per_arm_batch[p, arm_pulled[p]] += 1

        self.update_pulled_and_success(arm_pulled, visited_products, num_bought_products, num_primary, repeat=repeat)


    def update(self, price_pulled, reward, visited_products, n_bought_products):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward, visited_products, n_bought_products)
        if debug:
            print("PRICE PULLED : ", price_pulled, end="")
            print(" REWARD OBSERVED : ", np.sum(reward))

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

    def get_opt_arm_value(self):
        """
        :return: for every product choose the arm to pull
        :rtype: list
        """
        idx = [0 for _ in range(self.n_products)]
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = np.random.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            # arm of the current product with highest expected reward
            nearby_reward = np.ones(self.n_arms)
            if not np.all(self.nearby_reward[prod] == np.zeros(self.n_arms)):
                nearby_reward = self.nearby_reward[prod]
            idx[prod] = np.max(beta * ((get_all_margins()[prod] * self.users[0].get_n_items_to_buy(prod)) + nearby_reward))
        return idx

    def update_for_all_arms(self):

        self.beta_parameters[:, :, 0] = self.beta_parameters[:, :, 0] + self.success_per_arm_batch[:, :]
        self.beta_parameters[:, :, 1] = self.beta_parameters[:, :, 1] \
                                        + self.pulled_per_arm_batch - self.success_per_arm_batch


        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))

        self.nearbyReward = np.zeros((self.n_products, self.n_arms))
        self.visit_probability_estimation = np.zeros((5, 5))

        self.nearbyReward[np.isnan(self.nearbyReward)] = 0

        self.average_reward.append(np.mean(self.current_reward[-self.interaction:]))

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

    def isUcb(self):
        return False