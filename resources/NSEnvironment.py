import copy

from resources.Environment import *
from scipy.stats import beta

class NSEnvironment(Environment):
    def __init__(self, n_prices, prices, margins, lamb, secondary, prices_index, user_classes, users, changes_instant, same_user_saved_behaviour=None):
        super().__init__(n_prices, prices, margins, lamb, secondary, prices_index, user_classes, users)
        self.t = 0
        self.changes_instant = changes_instant  # list of instant iteration values of abrupt change

        # collect user (of each classes with updated convs)
        self.changes_collector = [(0,copy.deepcopy(self.users))]
        self.same_user_saved_behaviour = same_user_saved_behaviour

    def round(self, pulled_arm):
        self.t += 1
        if self.t in self.changes_instant:
            print(" ITS TIME TO ABRUPT CHANGE !!!")
            if self.same_user_saved_behaviour:
                for t, user in self.same_user_saved_behaviour:
                    if t == self.t:
                        new_user = copy.deepcopy(user)
                        self.users = new_user
                        self.changes_collector.append((self.t, new_user))
                        continue
            else :
                if self.changes_instant.index(self.t)%2 == 0:
                    self._abrupt_change_with_changes_collector()
                else:
                    self._abrupt_change1()
            print(self.changes_collector)

        return super().round(pulled_arm)

    # Method for step 6
    def round_6(self, pulled_arm):
        self.t += 1
        return super().round(pulled_arm)

    def _abrupt_change1(self):

        classes_idx = [i for i in range(len(self.users))]
        # Define a maximum value of conversion rates for every user class
        for i in range(len(classes_idx)):
            min_demand = classes[classes_idx[i]]["demand"]["min_demand"]
            max_demand = classes[classes_idx[i]]["demand"]["max_demand"]
            conv_rates = npr.uniform(min_demand, max_demand, (numbers_of_products, different_value_of_prices))
            self.users[i].conv_rates = conv_rates

        self.changes_collector.append((self.t, copy.deepcopy(self.users)))

    def _abrupt_change_with_changes_collector(self):
        classes_idx = [i for i in range(len(self.users))]
        # Define a maximum value of conversion rates for every user class
        print("PAST CONV RATES : \n", self.users[0].conv_rates)
        for i in range(len(classes_idx)):
            randoms = npr.uniform(0.01, 0.3, (numbers_of_products, different_value_of_prices))
            conv_rates = -np.sort(-randoms)
            self.users[i].conv_rates = conv_rates
        print("NEW CONV RATES : \n",self.users[0].conv_rates)
        self.changes_collector.append((self.t, copy.deepcopy(self.users)))

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
