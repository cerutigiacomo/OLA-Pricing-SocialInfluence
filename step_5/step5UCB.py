from Learner.clairvoyant import *
from plotting.plot_reward_regret import plot_regret_reward
from resources.Environment import Environment
from step_3.iterate_env import iterate, compute_ratio_regret


def enumerate_price_products(conv_rate, wdt):
    enumeration_of_triples = []
    # trple of values (product,price,expected_reward_SCALED,confidence)
    for i in range(0, numbers_of_products):
        for j in range(0, learner.n_arms):
            enumeration_of_triples.append((i, j, conv_rate[i, j], wdt[i, j]))
    return enumeration_of_triples


f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]
debug = False
fixed_maximum = 15
class_choosed = [0]

prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
# TODO : update aggregate data which has been taken from Student for test purposes
users = get_users(class_choosed)
conv_rates_aggregated = users[0].conv_rates

#
clairvoyant_price_index, clairvoyant_margin = find_clairvoyant_indexes(conv_rates_aggregated)


######### UCB
# TODO iterate the learner more times and get the mean of the results

learner = UCBLearner(lamb, secondary, [0], 4, step = 5)
# conversion_rates not observable, then the learner will estimate them.
#for i in range(len(class_choosed)):
#    learner.users[i].conv_rates = npr.rand(numbers_of_products, different_value_of_prices)
#    learner.users[i].alpha = npr.dirichlet(npr.random(numbers_of_products+1), 1).reshape(numbers_of_products + 1)
#    learner.users[i].max_item_bought = npr.random(1) * fixed_maximum

iteration = 100
daily_interaction = 50

#final_reward= np.zeros(iteration)
#cumulative_regret = np.zeros(iteration)
#cumulative_reward = np.zeros(iteration)

env = Environment(different_value_of_prices, prices, margins, lamb, secondary, [0, 0, 0, 0, 0], class_choosed, get_users(class_choosed))
cumulative_regret1_, cumulative_reward1, final_reward1, clairvoyant_margin_values1 = iterate(learner, env, iteration, daily_interaction, clairvoyant_price_index, "step5UCB")

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

repetition = final_reward1.shape[0]

clairvoyant_margin_values = np.tile(clairvoyant_margin_values1,(repetition,1))
cumulative_clairvoyant = np.cumsum(clairvoyant_margin_values, axis=1)

cumulative_regret1 = np.subtract(cumulative_clairvoyant, cumulative_reward1)

name_alg1 = "step5UCB"
color_alg1 = "blue"

plot_regret_reward(cumulative_regret1,
                   cumulative_reward1,
                   final_reward1,
                   clairvoyant_margin_values,
                   changes=[],
                   label_alg=name_alg1,
                   color_alg=color_alg1,
                   ax=axs,
                   day=iteration)

for ax in axs:
    ax.legend(loc='upper right')
    ax.grid()
plt.show()

ratio1 = compute_ratio_regret(learner, clairvoyant_margin_values, final_reward1, cumulative_regret1)

print(ratio1)
