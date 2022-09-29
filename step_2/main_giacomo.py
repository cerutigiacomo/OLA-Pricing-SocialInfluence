from users import *
from greedyReward import *
import matplotlib.pyplot as plt

# Objective function is the maximization of the cumulative
# expected margin over all the products

# Design a greedy algorithm to optimize the objective function
# when all the parameters are known.

alpha_ratios = npr.dirichlet([30,10,10,10,10,30])
total_users = npr.normal(500,10)
graph_weights = npr.random((5, 5))

v = np.array([3, 3, 2, 1]).reshape((4, 1))
n_items_bought = np.int64(npr.normal(1, 0.2, (4, 5)) * np.hstack((v, v, v, v, v)))
v = np.array([0.8, 0.6, 0.4, 0.2]).reshape((4, 1))
conv_rates = npr.normal(1, 0.2, (4, 5)) * np.hstack((v, v, v, v, v))

margins = npr.random((4, 5)) * 20

# Order the margins.
margins = np.sort(margins, axis=0)
prices = [0, 0, 0, 0, 0]
lamb = 0.2

users_A = Users_group(total_users, alpha_ratios, graph_weights, n_items_bought, conv_rates)

greedy = GreedyReward(prices, margins, lamb, users_A)
# Get the margins at the lowest price.
best_margins = greedy.bestReward()
print(greedy.prices, "BEST MARGIN INDEX")
print(best_margins, "\nBEST MARGIN")
print(np.sum(best_margins), "  COMULATIVE MARGIN")
# Plot the margins and rewards
d = greedy.list_prices.copy()
m = greedy.list_margins.copy()

plt.figure(0)
plt.plot(d, m, label="rewards decision")
plt.xlabel("price")
plt.xticks(rotation='vertical')
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.25)
plt.ylabel("margin")

plt.legend()
plt.show()
