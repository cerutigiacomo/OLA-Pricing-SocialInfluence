from greedyReward import *
from plotting.plot_distributions import *
from resources.define_distribution import *
from plot_greedy import *
import json

debug_print_distribution = True
f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]



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
users = get_users(classes_idx)

# Plot distributions
if debug_print_distribution:
    plot_simulator(sim)
    plot_users(users, classes_idx)

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
