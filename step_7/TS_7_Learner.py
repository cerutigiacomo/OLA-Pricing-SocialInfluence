import numpy as np

from step_7.Learner import *

reduction_factor = 1 # used to reduce the number of customer during the learner simulation for updating the means

class TSLearner(Learner):

    def __init__(self, users_classes, n_prices, interaction, n_products=numbers_of_products, step=4):
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

        self.success_by_arm = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm = np.zeros((self.n_products, self.n_arms))
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
            nearby_reward = np.ones(self.n_arms)
            if not np.all(self.nearby_reward[prod] == np.zeros(self.n_arms)):
                nearby_reward = self.nearby_reward[prod]
            a = np.argmax(beta * (get_all_margins()[prod] * self.users[0].get_n_items_to_buy(prod) * nearby_reward))
            index[prod] = a
            #idx[prod] = np.argmax(beta * ((self.prices[prod] * self.num_product_sold_estimation[prod]) + self.nearbyReward[prod]))
        return index

    def update_pulled_and_success(self, pulled_arm, visited, n_bought_products, items_rewards):
        for i in range(len(visited)):
            for j in range(len(visited[i])):
                prod = visited[i][j]
                if n_bought_products[i][prod] > 0:
                    self.success_by_arm[prod, pulled_arm[prod]] += 1
                self.pulled_per_arm[prod, pulled_arm[prod]] += 1
        # Conversion rate per product and price (matrix!)
        # Estimate the sample_conversion rate for each product
        counters = np.zeros(shape=self.n_products)
        estimed_conv_rate = self.estimed_conv_rate
        for (i, (visited_i, bought_i, rewards_i)) in enumerate(
                zip(visited, n_bought_products, items_rewards)):
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

            self.estimed_conv_rate = estimed_conv_rate

            self.update_users_conv_rates()

        self.users = \
            update_step_parameters_of_simulation(self.users, estimed_conv_rate, visited, n_bought_products,
                                                 n_step=4)

    def update(self, price_pulled, reward, visited_products, num_bought_products):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward, visited_products, num_bought_products)
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



    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = get_prices_and_margins(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward, a, b, c = website_simulation(self.sim, self.users, reduction_factor)
        return observed_reward

    def debug(self):
        if debug:
            print("beta par : \n", self.beta_parameters)


    def updateHistory(self, reward, arm_pulled, visited_products, num_bought_products, num_primary=None):
        super().update(arm_pulled, reward, visited_products, num_bought_products)

        num_bought_p = np.zeros(numbers_of_products)
        for i in range(len(num_bought_products[0])):
            for j in range(self.n_products):
                num_bought_p[j] += num_bought_products[0][i][j]

        if not isinstance(arm_pulled[0], list):
            current_prices = [i[j] for i, j in zip(get_all_margins(), arm_pulled)]
            current_reward = sum(num_bought_p * np.array(current_prices))
        else:
            if len(arm_pulled[0]) == 0:
                return
            current_prices = [i[j] for i, j in zip(get_all_margins(), arm_pulled[0][-1])]
            current_reward = sum(num_bought_products[0][-1] * np.array(current_prices))


        self.current_reward.append(current_reward)

        for iter in range(len(num_bought_products[0])):
            for prod in range(self.n_products):
                if np.all([len(visited_products[0][z]) == 5 for z in range(len(visited_products[0]))]):
                    if visited_products[0][iter][prod] > 0:
                        if num_bought_products[0][iter][prod] > 1:
                            self.success_per_arm_batch[prod, arm_pulled[0][iter][prod]] += 1
                        self.pulled_per_arm_batch[prod, arm_pulled[0][iter][prod]] += 1
                else:
                    if prod in visited_products[0][iter]:
                        if num_bought_products[0][iter][prod] > 1:
                            self.success_per_arm_batch[prod, arm_pulled[prod]] += 1
                        self.pulled_per_arm_batch[prod, arm_pulled[prod]] += 1


    def nearby_reward_evaluation(self, price_pulled):
        # get the reward for each arm estimated by the simulation

        nearby_reward = np.zeros((self.n_products, self.n_arms))
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                price_index = price_pulled.copy()
                price_index[prod] = arm
                nearby_reward[prod][arm] = self.simulate(price_index)[prod]
        self.nearby_reward = (self.nearby_reward + nearby_reward) / 2
        #if debug:
        print("nearby_reward: ", self.nearby_reward)

    """
    Update the conversion rate of the users with the mean of the non zero estimated ones.
    """
    def update_users_conv_rates(self):
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                if self.estimed_conv_rate[prod][arm] != 0:
                    self.users[0].conv_rates[prod][arm] = self.estimed_conv_rate[prod][arm]

    def isUcb(self):
        return False