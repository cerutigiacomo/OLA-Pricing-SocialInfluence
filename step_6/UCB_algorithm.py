import numpy as np
import math
from Observer import *

class UCB_algorithm(Observer):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # used to count the number of time an arm was selected
        self.numbers_of_selections = [0] * n_arms
        self.empirical_mean = [0] * n_arms

    def pull_arm(self):

        pulled_arm = 0
        max_upper_bound = 0

        for arm in range(0, self.n_arms):
            # if the arm arm has been already pulled once
            if self.numbers_of_selections[arm] > 0:
                total_counts = sum(self.numbers_of_selections)
                bound_length = math.sqrt(0.01 * math.log(total_counts) / float(self.numbers_of_selections[arm]))
                upper_bound = self.empirical_mean[arm] + bound_length
            else:
                upper_bound = 1e400

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                pulled_arm = arm

        return pulled_arm

    def update(self, pulled_arm, reward):

        self.t += 1
        self.update_rewards(pulled_arm, reward)

        self.numbers_of_selections[pulled_arm] = self.numbers_of_selections[pulled_arm] + 1
        n = self.numbers_of_selections[pulled_arm]

        old_empirical_mean = self.empirical_mean[pulled_arm]
        self.empirical_mean[pulled_arm] = ((n - 1) / float(n)) * old_empirical_mean + (1 / float(n)) * reward


# for n in range(0, 1000):
#     ad = 0
#     max_upper_bound = 0
#
#     # As we don't have any prior knowledge about the selection of each ad,
#     # we will take the first 10 rounds as trial rounds. So, we set the if
#     # condition:  if (numbers_of_selections[i] > 0) so that the ads are
#     # selected at least once before entering into the main algorithm.
#     for i in range(0, d):
#         if (numbers_of_selections[i] > 0):
#             average_reward = sums_of_rewards[i] / numbers_of_selections[i]
#             delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
#             upper_bound = average_reward + delta_i
#         else:
#             upper_bound = 1e400
#             # Here we applied a trick in the else condition by taking
#             # the variable upper_bound to a huge number. This
#             # is because we want the first 10 rounds as trial rounds
#             # where the 10 ads are selected at least once. This trick
#             # will help us to do so.
#
#         if upper_bound > max_upper_bound:
#             max_upper_bound = upper_bound
#             ad = i
#     # used to append the different types of ads selected in each round
#     ads_selected.append(ad)
#     # used to count the number of time an ad was selected
#     numbers_of_selections[ad] = numbers_of_selections[ad] + 1
#     #!!!!!!!!
#     reward = dataset.values[n, ad]
#     sums_of_rewards[ad] = sums_of_rewards[ad] + reward
#     total_reward = total_reward + reward

# plt.hist(ads_selected)
# plt.title('Histogram of ads selections')
# plt.xlabel('Ads')
# plt.ylabel('Number of times each ad was selected')
# plt.show()