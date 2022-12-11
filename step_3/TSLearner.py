from Learner import *

class TSLearner(Learner):

    def __init__(self, lamb, secondary, users, n_prices, n_products=numbers_of_products):
        super().__init__(lamb, secondary, users, n_prices, n_products)

        # upper confidence bounds of arms
        # optimistic estimation of the rewards provided by arms
        self.beta_parameters = np.ones((n_products, n_prices, 2))
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
        self.beta_parameters[:, price_pulled, 0] = 1 #self.beta_parameters[:, price_pulled, 0] + reward
        self.beta_parameters[:, price_pulled, 1] = 0.1 #self.beta_parameters[:, price_pulled, 1] + 1.0 - reward
    def simulate(self, price_pulled):
        self.sim.prices, self.sim.margins = get_prices_and_margins(price_pulled)
        self.sim.prices_index = price_pulled
        observed_reward, a, b, c = website_simulation(self.sim, self.users)
        return observed_reward

    def debug(self):
        if debug:
            print("beta par : \n", self.beta_parameters)
