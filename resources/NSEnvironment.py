from resources.Environment import *


class NSEnvironment(Environment):

    def __init__(self, n_prices, prices, margins, lamb, secondary, prices_index, users, changes_instant):

        super().__init__(n_prices, prices, margins, lamb, secondary, prices_index, users)
        self.t = 0
        self.changes_instant = changes_instant

        # collect user (of each classes with updated convs)
        self.changes_collector = [self.users]

    def round(self, pulled_arm):
        self.t += 1
        # if self.t in self.changes_instant:
        #     print(" ITS TIME TO ABRUPT CHANGE !!!")
        #     self._abrupt_change()
        #     print(self.changes_collector)
        # launch website_simulation
        return super().round(pulled_arm)

    def _abrupt_change(self):
        # Random conversion rate (demand curve) generating number between [0, 1)
        # one matrix simulating a class of users subjected to abrupt changes
        changed_conv_rates = np.array([[0.143, 0.125, 0.765, 0.999],
                                       [0.123, 0.224, 0.934, 0.234],
                                       [0.234, 0.987, 0.593, 0.077],
                                       [0.221, 0.876, 0.265, 0.123],
                                       [0.721, 0.234, 0.013, 0.376]])

        self.users[0].conv_rates = changed_conv_rates


        print(" NEW CONVERSION RATE ARE: \n", changed_conv_rates)


    def set_margins(self, margin_index):
        super().set_margins(margin_index)
