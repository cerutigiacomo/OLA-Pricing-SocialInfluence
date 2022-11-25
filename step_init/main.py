from simulator import *
from users import *
from website_simulation import website_simulation
from plotting.plot_distributions import *
from resources.define_distribution import *
import json
import matplotlib.pyplot as plt

debug_print_distribution = True
f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]

def simple_run():
    reward = np.zeros(numbers_of_products)
    for i in range(users_classes):
        reward += website_simulation(sim, users[i])
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("total revenue ", reward)

    plot_reward(reward)


def simulate_multiple_days(days):
    plt.rcParams["figure.figsize"] = (15,10)
    total_reward = np.zeros(numbers_of_products)
    for j in range(days):
        reward = np.zeros(numbers_of_products)
        for i in range(users_classes):
            reward += website_simulation(sim, users[i])
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("total revenue ", reward)
        plt.plot(list(range(numbers_of_products)), reward, 'o-', mfc='none')
        total_reward += reward
        # Change the prices, alpha, total_users daily
        new_alpha = distribute_alpha()
        new_total_users = distribute_total_user()
        for i in range(users_classes):
            users[i].alpha = new_alpha[i]
            users[i].total_users = new_total_users[i]
        sim.prices = distribute_prices()


    plt.xlabel('Product')
    plt.ylabel('Reward')
    plt.xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
    plt.grid()
    plt.suptitle('Reward for single day')
    plt.show()

    plot_reward(total_reward)


# DEFINE THE SIMULATOR
prices, margins, secondary = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
sim = Simulator(prices, margins, lamb, secondary)

# DEFINE 3 CLASS OF USERS
total_users, alpha_ratios, graph, n_items_bought, conv_rates = user_distribution()

users = [Users_group(total_users[i], alpha_ratios[i], graph[i], n_items_bought[i], conv_rates[i])
         for i in range(users_classes)]

# Plot distributions
if debug_print_distribution:
    plot_simulator(margins, prices, secondary)
    plot_users(total_users, alpha_ratios, graph, n_items_bought, prices, max_item_bought, conv_rates)

# RUN the simulation
days = data["simulator"]["days"]
simulate_multiple_days(days)
