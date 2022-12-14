from resources.Environment import *


class NSEnvironment(Environment):
    def __init__(self, n_prices, prices, margins, lamb, secondary, prices_index, users, changes_instant):
        super().__init__(n_prices, prices, margins, lamb, secondary, prices_index, users)
        self.t = 0
        self.changes_instant = changes_instant  # list of instant iteration values of abrupt change

        # collect user (of each classes with updated convs)
        self.changes_collector = [self.users]

    def round(self, pulled_arm):
        self.t += 1
        if self.t in self.changes_instant:
            print(" ITS TIME TO ABRUPT CHANGE !!!")
            self._abrupt_change()
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

    def _abrupt_change(self):
        classes_idx = [i for i in range(len(self.users))]
        # Define a maximum value of conversion rates for every user class
        for i in range(len(classes_idx)):
            conv_rates = npr.uniform(0.0, 1.0, (numbers_of_products, different_value_of_prices))
            self.users[i].conv_rates = conv_rates

        self.changes_collector.append((self.t, self.users))

    def set_margins(self, margin_index):
        super().set_margins(margin_index)
