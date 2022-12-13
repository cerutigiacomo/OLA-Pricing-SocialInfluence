from simulator import *
from users import *
from resources.define_distribution import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from plotting.plot_reward_regret import *
from step_3.clairvoyant import *
from resources.Environment import Environment
from iterate import iterate
from UCBLearner import *


def enumerate_price_products(conv_rate, wdt):
    enumeration_of_triples = []
    # trple of values (product,price,expected_reward_SCALED,confidence)
    for i in range(0, numbers_of_products):
        for j in range(0, learner.n_arms):
            enumeration_of_triples.append((i, j, conv_rate[i, j], wdt[i, j]))
    return enumeration_of_triples


# method  called when RandomState is initialized. It can be called again to re-seed the generator.
np.random.seed(10)

f = open('../resources/environment.json')
data = json.load(f)
different_value_of_prices = data["product"]["different_value_of_prices"]
numbers_of_products = data["product"]["numbers_of_products"]
classes = data["users"]["classes"]
users_classes = len(classes)
prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
# TODO : update aggregate data which has been taken from Student for test purposes
debug = False
class_choosed = [0]
users = get_users(class_choosed)

classes_idx = [0]
conv_rates = np.zeros((len(classes_idx), numbers_of_products, different_value_of_prices))
for i in range(len(classes_idx)):
    min_demand = classes[classes_idx[i]]["demand"]["min_demand"]
    max_demand = classes[classes_idx[i]]["demand"]["max_demand"]
    # Define a maximum value of conversion rates for every user class
    conv_rates[i] = npr.uniform(min_demand, max_demand, (numbers_of_products, different_value_of_prices))

#print ("INITIAL CONV_RATES:", conv_rates[0], "\n RANDOM CONV RATES:", changed_conv_rates)

clairvoyant_price_index, clairvoyant_margin_values = find_clairvoyant_indexes(conv_rates)

# Args: lambda, secondary connected product,
# [0] user in this case 'student', number of prices
learner = UCBLearner(lamb, secondary, [0], 4)

iteration = 100
daily_interaction = 20

# [0 0 0 0 0] is the prices_index
env = Environment(different_value_of_prices, prices, margins, lamb, secondary, [0, 0, 0, 0, 0], class_choosed)
iterate(learner, env, iteration, daily_interaction, clairvoyant_price_index, "step3_UCB", 3)

rewards = learner.means
widths = learner.widths

pp = enumerate_price_products(rewards,widths)


fig2 = plt.figure(2,figsize=(30,10))
x_values = np.arange(len(pp))
colors = ["r","g","b","y","m"]
for x in x_values:
    plt.scatter(x,pp[x][2],color=colors[pp[x][0]])
    #plt.vlines(x=x,ymin=pp[x][2]-pp[x][3],ymax=pp[x][2]+pp[x][3],colors=colors[pp[x][0]])
    plt.errorbar(x=x,y=pp[x][2],yerr=pp[x][3],color=colors[pp[x][0]],capsize=3)
plt.xticks(x_values)
plt.ylim(-2, 3)
plt.grid()
plt.show()