from Learner import *

class TSLearner(Learner):

    def __init__(self, lamb, secondary, users_classes, n_prices, clairvoyant_margin_values, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users_classes, n_prices, n_products)

        # conversion_rates not observable
        for i in range(len(users_classes)):
            self.users[i].conv_rates = npr.rand(numbers_of_products,different_value_of_prices)
        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.beta_parameters = np.ones((n_products, n_prices, 2))
        self.clairvoyant_margin_values = np.mean(clairvoyant_margin_values)
        # load all the margins
        self.margin = get_all_margins()

    def act(self):
        idx = [0 for _ in range(self.n_products)]
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = npr.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            idx[prod] = np.argmax(beta * (self.margin[prod]))
        return idx

    def update(self, price_pulled, reward):
        # MAIN UPDATE FOR RESULTS PRESENTATION
        super().update(price_pulled, reward)
        if debug:
            print("PRICE PULLED : \n", price_pulled)
            print("REWARD OBSERVED : \n", reward)

        # TODO update beta parameters
        # get the reward for each arm estimated by the simulation
        reward_2 = np.zeros(range(self.n_products))
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                arm_index = np.zeros(self.n_products).astype(int)
                arm_index[prod] = int(arm)
                reward_2 = self.simulate(arm_index)
                if reward[prod] > 0:
                    self.beta_parameters[prod, arm, 0] += reward_2[prod]
                else:
                    self.beta_parameters[prod, arm, 1] -= reward_2[prod]

        reward_scaled = self.get_reward_scaled(reward_2)
        self.beta_parameters[:, price_pulled, 0] = self.beta_parameters[:, price_pulled, 0] + reward_scaled
        self.beta_parameters[:, price_pulled, 1] = self.beta_parameters[:, price_pulled, 1] + 1.0 - reward_scaled

    def get_reward_scaled(self, reward):
        return reward


    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = get_prices_and_margins(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward, a, b, c = website_simulation(self.sim, self.users)
        return observed_reward

    def debug(self):
        if debug:
            print("beta par : \n", self.beta_parameters)
