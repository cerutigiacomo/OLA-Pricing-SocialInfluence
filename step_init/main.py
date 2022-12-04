from simulator import *
from users import *
from website_simulation import *
from plotting.plot_distributions import *
from resources.define_distribution import *
import json

debug_print_distribution = True
f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]

def simple_run():
    reward = website_simulation(sim, users)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("total revenue ", reward)

    plot_reward(reward)



# DEFINE THE SIMULATOR
prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
sim = Simulator(prices, margins, lamb, secondary, [today for _ in range(5)])

# DEFINE 3 CLASS OF USERS
classes_idx = [i for i in range(users_classes)]
total_users, alpha_ratios, graph, n_items_bought, conv_rates, features = user_distribution(classes_idx)

users = [Users_group(total_users[i], alpha_ratios[i], graph[i], n_items_bought[i], conv_rates[i],features[i])
         for i in range(users_classes)]

# Plot distributions
if debug_print_distribution:
    plot_simulator(margins, prices, secondary)
    plot_users(total_users, alpha_ratios, graph, n_items_bought, max_item_bought, conv_rates, classes_idx)

# RUN the simulation
days = data["simulator"]["days"]
simulate_multiple_days(sim, users, classes_idx)
