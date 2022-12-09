from users import *
from greedyReward import *
from plotting.plot_distributions import *
from resources.define_distribution import *
from plot_greedy import *
import json

debug_print_distribution = False
f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]

def get_all_margins():
    data_ = [data["product"]["products"][i] for i in range(5)]
    margin = np.zeros((4,5))
    for j in range(4):
        margin[j] = np.array([data_[i]["price"][j] - data_[i]["cost"] for i in range(5)])
    return margin.transpose()

def clairvoyant_sol():
    conv_prices = (get_all_margins() * users[0].conv_rates)
    indexes = np.argmax(conv_prices, axis=1)
    #a = [1, 0, 1, 1, 1]
    sim.prices, sim.margins = get_prices_and_margins(indexes)
    reward = np.zeros((5,5))
    for i in range(5):
        reward[i], a, b, c = website_simulation(sim, users)
    rew = np.max(reward, axis=0)
    return rew, indexes



# Objective function is the maximization of the cumulative expected margin over all the products

# Design a greedy algorithm to optimize the objective function
# when all the parameters are known.
# DEFINE THE SIMULATOR
prices_greedy, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
sim = Simulator(prices_greedy, margins, lamb, secondary, [today for _ in range(5)])

# DEFINE 1 CLASS OF USERS, the 1st one
classes_idx = [0]
total_users, alpha_ratios, graph, n_items_bought, conv_rates, features = user_distribution(classes_idx)

users = [Users_group(total_users[i], alpha_ratios[i], graph[i], n_items_bought[i], conv_rates[i], features[i])
         for i in range(len(classes_idx))]

# Plot distributions
if debug_print_distribution:
    plot_simulator(margins, prices_greedy, secondary)
    plot_users(total_users, alpha_ratios, graph, n_items_bought, max_item_bought, conv_rates, classes_idx)

reward_sol, b_m = clairvoyant_sol()

greedy = GreedyReward(lamb, secondary, users, [0])
# Get the margins at the lowest price.
best_margins = greedy.bestReward()
prices_greedy = greedy.list_prices.copy()
margin_greedy = greedy.list_margins.copy()
# Plot the margins and rewards
plot_greedy(prices_greedy, margin_greedy)

print("Clairvoyant sol: ", sum(reward_sol), "indexes: ", b_m)
print("Greedy sol: ", str(margin_greedy[-1]), "indexes: ", prices_greedy[-1])

plot_reward_comparison(["Clairvoyant" + str(b_m), "Greedy" + str(prices_greedy[-1])], [sum(reward_sol), margin_greedy[-1]])
