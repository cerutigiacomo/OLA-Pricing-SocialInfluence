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
        idx = np.array([0 for _ in range(self.n_products)])
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = npr.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            idx[prod] = np.argmax(beta * (self.margin[prod]))
        return idx

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

        # TODO update beta parameters
        self.beta_parameters[np.arange(self.n_products), price_pulled, 0] = \
            self.beta_parameters[np.arange(self.n_products), price_pulled, 0] + sample_conv_rates
        self.beta_parameters[np.arange(self.n_products), price_pulled, 1] = \
            self.beta_parameters[np.arange(self.n_products), price_pulled, 1] + 1.0 - sample_conv_rates

    def debug(self):
        if debug:
            print("beta par : \n", self.beta_parameters)
