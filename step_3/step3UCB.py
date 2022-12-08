from simulator import Simulator
from users import *
from plotting.plot_distributions import *
from resources.define_distribution import *
import json
import matplotlib.pyplot as plt
from step_3.Learner import UCBLearner

f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]

prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
sim = Simulator(prices, margins, lamb, secondary, [today for _ in range(5)])

total_users, alpha_ratios, graph, n_items_bought, conv_rates, features = user_distribution()

# TODO : update aggregate data which has been taken from Student for test purposes
users = [Users_group(total_users[0], alpha_ratios[0], graph[0], n_items_bought[0], conv_rates[0], features[0])]

# CLAIROVOYANT
# TODO : again i have the Student for test purposes
conv_rates_aggregated = conv_rates[0]

clairvoyant = np.argmax(conv_rates_aggregated, axis=1)
print(conv_rates_aggregated)

def find_clairvoyant(conv_rates_aggregated):

    prices_of_products = []
    costs_of_products = []
    products = get_product()
    for i in range(len(products)):
        prices_of_products.append(products[i]['price'])
        costs_of_products.append(products[i]['cost'])
    prices_of_products = np.array(prices_of_products)
    costs_of_products = np.array(costs_of_products)

    print(prices_of_products)
    print(costs_of_products)

    costs_of_products = costs_of_products.reshape(1,-1)
    margins_of_products = np.subtract(prices_of_products,np.tile(costs_of_products.transpose(),(1,4)))

    print(margins_of_products)

    # BEST MARGIN FOR A SINGLE UNIT OF PRODUCT
    print(np.multiply(conv_rates_aggregated, margins_of_products))
    best_margin_for_unit = np.argmax(np.multiply(conv_rates_aggregated,margins_of_products),axis=1)
    clairvoyant_margin_values = margins_of_products[np.arange(numbers_of_products),best_margin_for_unit]

    return best_margin_for_unit, clairvoyant_margin_values

clairvoyant_price_index, clairvoyant_margin_values = find_clairvoyant(conv_rates_aggregated)
print("Indici dei clairvoyant price_index : \n",clairvoyant_price_index)
print("Effettivi valori di margine : \n", clairvoyant_margin_values)
#print("Massimo numero di items comprati  = ", n_items_aggregated)


######### UCB

learner = UCBLearner(lamb, secondary, users, 4, [0])

#
for iterations in range(100):
    learner.debug()
    price_pulled = learner.act()
    reward_observed = learner.simulate(price_pulled)
    learner.update(price_pulled, reward_observed)

x_labels = learner.list_prices
y = learner.list_margins

x_values = [i for i in range(x_labels.shape[0])]
print(x_labels,y)

fig = plt.figure(figsize=(30, 10))
plt.plot(x_values,y)
plt.xticks(ticks=x_values,labels=x_labels,rotation=90)

######## clairvoyant

# TODO : calirvoyant solution
# a first attempt and reported below is to simply simulate with the best price setting of clairvoyant
y_clairvoyant = np.sum(learner.simulate(clairvoyant_price_index))
# a second attempt could be to use a different simulator (for example by setting it to the higher possbile n_boughts)
# but i have doubt about how to handle the graph secondaries, and alpha ratios, ...
# NOTE : running the algorithm for 1000 or more iterations shows the limit of this clairvoyant solution

plt.plot(x_values,np.repeat(y_clairvoyant,len(x_values)),"r-")

plt.show()
