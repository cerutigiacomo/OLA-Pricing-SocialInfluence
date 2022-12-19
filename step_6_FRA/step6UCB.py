import copy

from Learner.Learner import *
from Learner.clairvoyant import find_clairvoyant_indexes
from plotting.plot_reward_regret import plot_regret_reward
from resources.NSEnvironment import NSEnvironment
from step_3.UCBLearner import UCBLearner
from step_3.iterate_env import iterate, compute_ratio_regret
from step_6_FRA.UCBSWLearner import UCBSWLearner


def enumerate_price_products(conv_rate, wdt):
    enumeration_of_triples = []
    # trple of values (product,price,expected_reward_SCALED,confidence)
    for i in range(0, numbers_of_products):
        for j in range(0, different_value_of_prices):
            enumeration_of_triples.append((i, j, conv_rate[i, j], wdt[i, j]))
    return enumeration_of_triples


f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]
debug = False
class_choosed = [0]

prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
# TODO : update aggregate data which has been taken from Student for test purposes
users = get_users(class_choosed)
conv_rates_aggregated = users[0].conv_rates

#
clairvoyant_price_index, clairvoyant_margin_values = find_clairvoyant_indexes(conv_rates_aggregated)


######### UCB
# TODO iterate the learner more times and get the mean of the results

iteration = 100
daily_interaction = 50

tau = int(np.sqrt(iteration))

name_alg1 = "step6SWUCB"
color_alg1 = "blue"
learner1 = UCBSWLearner(lamb, secondary, [0], 4, tau)

for i in range(len(class_choosed)):
    learner1.users[i].conv_rates = npr.rand(numbers_of_products, different_value_of_prices)

# INSTANT ABRUPT CHANGES
changes_instant = [50, 80]

users_init = copy.deepcopy(users)
env = NSEnvironment(different_value_of_prices, prices, margins, lamb, secondary, [0, 0, 0, 0, 0], class_choosed, users_init, changes_instant)
cumulative_regret1_, cumulative_reward1, final_reward1, clairvoyant_margin_values1 = iterate(learner1, env, iteration, daily_interaction, clairvoyant_price_index, name_alg1, 6)


# SETUP ENV FOR SECOND LEARNER
same_user_saved_behaviour = env.changes_collector
env = NSEnvironment(different_value_of_prices, prices, margins, lamb, secondary, [0, 0, 0, 0, 0], class_choosed, users, changes_instant, same_user_saved_behaviour)

name_alg2 = "step6UCB"
color_alg2 = "green"
learner2 = UCBLearner(lamb, secondary, [0], 4)
cumulative_regret2_, cumulative_reward2, final_reward2, clairvoyant_margin_values2 = iterate(learner2, env, iteration, daily_interaction, clairvoyant_price_index, name_alg2, 6)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

###
### unify clairvoyant
###

clairvoyants = np.vstack((clairvoyant_margin_values1,clairvoyant_margin_values2))
clairvoyant_margin_values = np.max(clairvoyants, axis=0)
repetition = final_reward1.shape[0]

clairvoyant_margin_values = np.tile(clairvoyant_margin_values,(repetition,1))
cumulative_clairvoyant = np.cumsum(clairvoyant_margin_values, axis=1)


cumulative_regret1 = np.subtract(cumulative_clairvoyant, cumulative_reward1)

plot_regret_reward(cumulative_regret1,
                   cumulative_reward1,
                   final_reward1,
                   clairvoyant_margin_values,
                   changes=changes_instant,
                   label_alg=name_alg1,
                   color_alg=color_alg1,
                   ax=axs,
                   day=iteration)


cumulative_regret2 = np.subtract(cumulative_clairvoyant, cumulative_reward2)

plot_regret_reward(cumulative_regret2,
                   cumulative_reward2,
                   final_reward2,
                   clairvoyant_margin_values,
                   changes=changes_instant,
                   label_alg=name_alg2,
                   color_alg=color_alg2,
                   ax=axs,
                   day=iteration)
for ax in axs:
    ax.legend(loc='upper left')
    ax.grid()
plt.show()

ratio1 = compute_ratio_regret(learner1, clairvoyant_margin_values, final_reward1, cumulative_regret1)
ratio2 = compute_ratio_regret(learner2, clairvoyant_margin_values, final_reward2, cumulative_regret2)

print("RATIOS :")
print(ratio1)
print(ratio2)