from users import *
from greedyReward import *
from website_simulation import website_simulation
from plotting.plot_distributions import *
from resources.define_distribution import *
import json
import matplotlib.pyplot as plt

debug_print_distribution = True
f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]


# Objective function is the maximization of the cumulative expected margin over all the products

# Design a greedy algorithm to optimize the objective function
# when all the parameters are known.

# DEFINE THE SIMULATOR
prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
sim = Simulator(prices, margins, lamb, secondary, [today for _ in range(5)])

# DEFINE 1 CLASS OF USERS, the 1st one
classes_idx = [0]
total_users, alpha_ratios, graph, n_items_bought, conv_rates, features = user_distribution(classes_idx)

users = [Users_group(total_users[i], alpha_ratios[i], graph[i], n_items_bought[i], conv_rates[i],features[i])
         for i in range(len(classes_idx))]

# Plot distributions
if debug_print_distribution:
    plot_simulator(margins, prices, secondary)
    plot_users(total_users, alpha_ratios, graph, n_items_bought, max_item_bought, conv_rates, classes_idx)


greedy = GreedyReward(lamb, secondary, users)
# Get the margins at the lowest price.
best_margins = greedy.bestReward()
d = greedy.list_prices.copy()
m = greedy.list_margins.copy()
print(d, "BEST MARGIN INDEX")
print(m, "\nBEST MARGIN")
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
plt.grid()

plt.legend()
plt.show()
