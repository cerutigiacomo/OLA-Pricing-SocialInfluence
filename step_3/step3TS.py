from TSLearner import *
from iterate_env import *
from resources.Environment import *

f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]
debug = False
class_choosed = [0]

prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
users = get_users(class_choosed)
iteration = 100
daily_simulation = 20
daily_iteration_mean = 10


conv_rates_aggregated = users[0].conv_rates
clairvoyant_price_index, clairvoyant_margin_values = find_clairvoyant_indexes(conv_rates_aggregated)


######### TS
learner = TSLearner(lamb, secondary, class_choosed, different_value_of_prices, step=3)
env = Environment(different_value_of_prices, prices, margins, lamb, secondary,
                  [0, 0, 0, 0, 0], class_choosed, get_users(class_choosed),
                  daily_iteration_mean)

# conversion_rates not observable, then the learner will estimate them.
for i in range(len(class_choosed)):
    learner.users[i].conv_rates = npr.rand(numbers_of_products, different_value_of_prices)


iterate(learner, env, iteration, daily_simulation, clairvoyant_price_index, "step3TS")
