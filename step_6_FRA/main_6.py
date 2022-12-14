from simulator import *
from website_simulation import *
from UCB_SW_algorithm import *
from step_6_FRA.Environment.NS_Environment import *
from resources.define_distribution import *
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

classes_idx = [0]
conv_rates = np.zeros((len(classes_idx), numbers_of_products, different_value_of_prices))
for i in range(len(classes_idx)):
    min_demand = classes[classes_idx[i]]["demand"]["min_demand"]
    max_demand = classes[classes_idx[i]]["demand"]["max_demand"]
    # Define a maximum value of conversion rates for every user class
    conv_rates[i] = npr.uniform(min_demand, max_demand, (numbers_of_products, different_value_of_prices))
#print ("INITIAL CONV_RATES:", conv_rates)

# Random conversion rate (demand curve) generating number between [0, 1)
# one matrix simulating a class of users subjected to abrupt changes
changed_conv_rates = np.array( [[0.943, 0.125, 0.765, 0.999],
                      [0.123, 0.224, 0.934, 0.234],
                      [0.234, 0.987, 0.893, 0.677],
                      [0.321, 0.876, 0.765, 0.123],
                      [0.321, 0.234, 0.013, 0.876]] )

print ("INITIAL CONV_RATES:", conv_rates[0], "\n RANDOM CONV RATES:", changed_conv_rates)


horizon = 300
n_experiments = 100
# single: per every single experiment
swucb_single_reward = []
ucb_single_reward = []
window_size = int(np.sqrt(horizon))

users = get_users([0])
user = users[0]
alpha = user.alpha
# item to choose where simulate the UCB/SW_UCB algorithm
product = 0

for e in range(0, n_experiments):
    print(e)
    # set the UCB1 env
    sw_env = NS_Environment(n_arms, changed_conv_rates, horizon)
    ucb_learner = UCB_algorithm(n_arms)
    # set the Sliding Window UCB1 env
    swucb_env = NS_Environment(n_arms, changed_conv_rates, horizon)
    swucb_learner = UCB_SW_algorithm(n_arms, window_size)

    for t in range(0, horizon):
        pulled_arm = ucb_learner.pull_arm()
        print ("PULLED ARM:", pulled_arm)
        # Bernoulli result of pulled arm
        result = sw_env.round(pulled_arm)
        reward = ucb_learner.simulation(product, user, changed_conv_rates, result)
        print("Reward UCB (step no.", e,")", reward)
        ucb_learner.update(pulled_arm, reward)

        pulled_arm = swucb_learner.pull_arm()
        result = swucb_env.round(pulled_arm)
        reward = swucb_learner.simulation(product,user,changed_conv_rates, result)
        print("Reward SW (step no.", e,")", reward)
        swucb_learner.update(pulled_arm, reward)
        print("------------------------------")

    ucb_single_reward.append(ucb_learner.total_rewards)
    swucb_single_reward.append(swucb_learner.total_rewards)

    if e == 50:
        print("ABRUPT CHANGE !!!")
        conv_rates[0] = changed_conv_rates

ucb_instantaneus_regret = np.zeros(horizon)
swucb_instantaneus_regret = np.zeros(horizon)
n_phases = len(changed_conv_rates)
phases_len = int(horizon / n_phases)
optimal_per_phases = changed_conv_rates.max(axis=1)

print("Abrupt changes produces this probabilities:\n", changed_conv_rates)
print("Optimal per phases:\n", optimal_per_phases)
opt_per_round = np.zeros(horizon)

for i in range(0, n_phases):
    opt_per_round[i * phases_len: (i + 1) * phases_len] = optimal_per_phases[i]
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
    classes_idx = [i for i in range(users_classes)]
    users = get_users(classes_idx)
    # TODO: maybe run the sim just once for a class with new demand curve,
    # instead of 3 times with the same curve



    max_item_bought = data["simulator"]["max_item_bought"]

    # Plot distributions
    plot_simulator(sim)
    plot_users(users, classes_idx)

    # RUN the simulation
    days = data["simulator"]["days"]
    website_simulation(sim, users)
    #simulate_multiple_days(sim, users)