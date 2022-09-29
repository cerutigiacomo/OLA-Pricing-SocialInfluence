class Learner:
    def __init__(self, n_products, n_prices):
        self.t = 0
        self.n_products = n_products
        self.n_prices = n_prices
        self.collected_rewards = []
        self.rewards = [[]for _ in range(n_prices)]*n_products

    def reset(self):
        self.__init__(self.n_products, self.n_prices)

    def act(self):
        pass

    def update(self, product, price_pulled, reward):
        self.t += 1
        self.rewards[product][price_pulled].append(reward)
