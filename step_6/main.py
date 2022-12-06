from simulator import *
from users import *
from website_simulation import *
from plotting.plot_distributions import *
from resources.define_distribution import *
from website_simulation import *
import json
from UCB_SW_algorithm import *
from NS_environment import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

# method  called when RandomState is initialized. It can be called again to re-seed the generator.
np.random.seed(10)
n_arms = 4

f = open('../resources/environment.json')
data = json.load(f)
different_value_of_prices = data["product"]["different_value_of_prices"]
numbers_of_products = data["product"]["numbers_of_products"]
classes = data["users"]["classes"]
users_classes = len(classes)


# Random conversion rate (demand curve) generating number between [0, 1)
# one matrix simulating a class of users subjected to abrupt changes
demand_curve = npr.rand(5, 4)

horizon = 350
n_experiments = 300
# single: per every single experiment
swucb_single_reward = []
ucb_single_reward = []
window_size = int(np.sqrt(horizon))


for e in range(0, n_experiments):
    print(e)
    # set the UCB1 env
    sw_env = NS_environment(n_arms, demand_curve, horizon)
    ucb_learner = UCB_algorithm(n_arms)
    # set the Sliding Window UCB1 env
    swucb_env = NS_environment(n_arms, demand_curve, horizon)
    swucb_learner = UCB_SW_algorithm(n_arms, window_size)

    for t in range(0, horizon):
        pulled_arm = ucb_learner.pull_arm()
        reward = sw_env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)

        pulled_arm = swucb_learner.pull_arm()
        reward = swucb_env.round(pulled_arm)
        swucb_learner.update(pulled_arm, reward)

    ucb_single_reward.append(ucb_learner.total_rewards)
    swucb_single_reward.append(swucb_learner.total_rewards)

ucb_instantaneus_regret = np.zeros(horizon)
swucb_instantaneus_regret = np.zeros(horizon)
n_phases = len(demand_curve)
phases_len = int(horizon / n_phases)
optimal_per_phases = demand_curve.max(axis=1)

print("Abrupt changes produces this probabilities:\n", demand_curve)
print("Optimal per phases:\n", optimal_per_phases)
optimal_per_round = np.zeros(horizon)

for i in range(0, n_phases):
    optimal_per_round[i * phases_len: (i + 1) * phases_len] = optimal_per_phases[i]
    ucb_instantaneus_regret[i * phases_len: (i + 1) * phases_len] = optimal_per_phases[i] - np.mean(ucb_single_reward, axis=0)[i * phases_len: (i + 1) * phases_len]
    swucb_instantaneus_regret[i * phases_len: (i + 1) * phases_len] = optimal_per_phases[i] - np.mean(swucb_single_reward, axis=0)[i * phases_len: (i + 1) * phases_len]


#In the first figure we show the reward
plt.subplot(1,2,1)
plt.title("Reward plot")
plt.xlabel('Horizon')
plt.ylabel('Reward')
plt.plot(np.mean(ucb_single_reward, axis=0), 'r')
plt.plot(np.mean(swucb_single_reward, axis=0), 'b')
#plt.plot(opt_per_round, '--k')
plt.legend(['UCB1','SW-UCB1','Optimum'])

#In the second plot we show the regret
plt.subplot(1,2,2)
plt.title("Regret plot")
plt.xlabel ('Horizon')
plt.ylabel('Regret')
plt.plot(np.cumsum(ucb_instantaneus_regret), 'r')
plt.plot(np.cumsum(swucb_instantaneus_regret), 'b')
plt.legend(['UCB1','SW-UCB1'])

plt.show()

# Boolean to start the simulation and plot the graphs
wanna_simulate = False

if wanna_simulate:
    # DEFINE THE SIMULATOR
    prices, margins, secondary, today = simulator_distribution()
    lamb = data["product"]["lambda"]  # LAMBDA
    sim = Simulator(prices, margins, lamb, secondary, [today for _ in range(5)])

    # DEFINE 3 CLASS OF USERS
    total_users, alpha_ratios, graph, n_items_bought, demand_curve = user_distribution()

    # TODO: maybe run the sim just once for a class with new demand curve,
    # instead of 3 times with the same curve

    # general class with abrupt demand changes
    users = [Users_group(total_users[i], alpha_ratios[i], graph[i], n_items_bought[i], demand_curve[i])
             for i in range(users_classes)]

    max_item_bought = data["simulator"]["max_item_bought"]

    # Plot distributions
    plot_simulator(margins, prices, secondary)
    plot_users(total_users, alpha_ratios, graph, n_items_bought, max_item_bought, demand_curve)

    # RUN the simulation
    days = data["simulator"]["days"]
    website_simulation(sim, users)
    #simulate_multiple_days(sim, users)