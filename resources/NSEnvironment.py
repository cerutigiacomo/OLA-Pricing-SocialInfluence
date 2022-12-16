from resources.Environment import *


class NSEnvironment(Environment):
    def __init__(self, n_prices, prices, margins, lamb, secondary, prices_index, users, changes_instant):
        super().__init__(n_prices, prices, margins, lamb, secondary, prices_index, users)
        self.t = 0
        self.changes_instant = changes_instant  # list of instant iteration values of abrupt change

        # collect user (of each classes with updated convs)
        self.changes_collector = [(0,self.users)]

    def reset(self):
        self.__init__(self.n_arms, self.prices, self.sim.margins, self.lam, self.secondary, self.sim.prices_index, self.users_indexes, self.changes_instant)

    def round(self, pulled_arm):
        self.t += 1
        if self.t in self.changes_instant:
            print(" ITS TIME TO ABRUPT CHANGE !!!")
            self._abrupt_change_with_changes_collector()
            print(self.changes_collector)

        return super().round(pulled_arm)

    def _abrupt_change1(self):

        classes_idx = [i for i in range(len(self.users))]
        # Define a maximum value of conversion rates for every user class
        for i in range(len(classes_idx)):
            min_demand = classes[classes_idx[i]]["demand"]["min_demand"]
            max_demand = classes[classes_idx[i]]["demand"]["max_demand"]
            conv_rates = npr.uniform(min_demand, max_demand, (numbers_of_products, different_value_of_prices))
            self.users[i].conv_rates = conv_rates

        self.changes_collector.append((self.t, self.users))

    def _abrupt_change_with_changes_collector(self):
        classes_idx = [i for i in range(len(self.users))]
        # Define a maximum value of conversion rates for every user class
        for i in range(len(classes_idx)):
            conv_rates = npr.uniform(0.0, 1.0, (numbers_of_products, different_value_of_prices))
            self.users[i].conv_rates = conv_rates

        self.changes_collector.append((self.t, self.users))

    def _abrupt_change(self):
        # Random conversion rate (demand curve) generating number between [0, 1)
        # one matrix simulating a class of users subjected to abrupt changes
        changed_conv_rates = np.array([[0.143, 0.125, 0.765, 0.999],
                                       [0.123, 0.224, 0.934, 0.234],
                                       [0.234, 0.987, 0.593, 0.077],
                                       [0.221, 0.876, 0.265, 0.123],
                                       [0.721, 0.234, 0.013, 0.376]])

        self.users[0].conv_rates = changed_conv_rates


    def set_margins(self, margin_index):
        super().set_margins(margin_index)
