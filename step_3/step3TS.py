from TSLearner import *
from iterate_env import *
from resources.Environment import *

f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]
debug = False


prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
users = get_users([0])
iteration = 10


conv_rates_aggregated = users[0].conv_rates
clairvoyant_price_index, clairvoyant_margin_values = find_clairvoyant_indexes(conv_rates_aggregated)


######### TS
learner = TSLearner(lamb, secondary, [0], different_value_of_prices, clairvoyant_margin_values)
env = Environment(different_value_of_prices, prices, margins, lamb, secondary, [0, 0, 0, 0, 0], users)



# Clairvoyant solution
clairvoyant_margin_values = find_clairvoyant_reward(learner, clairvoyant_price_index, iteration)

iterate(learner, env, iteration, clairvoyant_margin_values, "step3TS")
