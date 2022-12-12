from plotting.plot_reward_regret import *
from clairvoyant import *
from resources.Environment import Environment
from step_3.iterate_env import iterate


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

prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
# TODO : update aggregate data which has been taken from Student for test purposes
users = get_users([0])
conv_rates_aggregated = users[0].conv_rates

#
clairvoyant_price_index, clairvoyant_margin_values = find_clairvoyant_indexes(conv_rates_aggregated)

learner = UCBLearner(lamb, secondary, [0], 4)

######### UCB

iteration = 300

env = Environment(different_value_of_prices, prices, margins, lamb, secondary, [0, 0, 0, 0, 0], get_users([0]))

iterate(learner, env, iteration, clairvoyant_price_index, "step3UCB", n_step=3)
y_clairvoyant = find_clairvoyant_reward(learner, env, clairvoyant_price_index, iteration)

# Clairvoyant solution
clairvoyant_margin = y_clairvoyant

# Plot UCB Regret and Reward
clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin)
cumulative_reward = np.cumsum(learner.list_margins)
cumulative_regret = np.cumsum(clairvoyant_margin_iterated) - cumulative_reward
final_reward = learner.list_margins
plot_regret_reward(cumulative_regret,
                   cumulative_reward,
                   final_reward,
                   clairvoyant_margin,
                   label_alg="Step3UCB",
                   day=iteration)

conversion_rates = learner.means
widths = learner.widths
pp = enumerate_price_products(conversion_rates, widths)
fig2 = plt.figure(2, figsize=(30, 10))

x_values = np.arange(len(pp))
colors = ["r", "g", "b", "y", "m"]
for x in x_values:
    plt.scatter(x, pp[x][2], color=colors[pp[x][0]])
    # plt.vlines(x=x,ymin=pp[x][2]-pp[x][3],ymax=pp[x][2]+pp[x][3],colors=colors[pp[x][0]])
    plt.errorbar(x=x, y=pp[x][2], yerr=pp[x][3], color=colors[pp[x][0]], capsize=3)
plt.xticks(x_values)
plt.ylim(-2, 3)
plt.title("Estimated conversion rates with confidence bounds")
plt.grid()
plt.show()



# compare estimated conv rates and true conv rates
print("differences between conv rates and estimated conv rates : \n",
      np.subtract(conv_rates_aggregated, learner.means))





def find_max_rewards(usr, scdy):
    # this method is used to find max value to min-max scale rewards
    # min value is 0, since the product could be not sold
    # it is the optimistic (probabilities to 1 of buy,visit,..) reward

    # 1. get margin of each product for buying one item
    price_indexes, margin_values = find_clairvoyant_indexes(usr.conv_rates)
    n_items_bought = usr.n_items_bought

    # 2. get reward obtained for each primary product ++ and consider buying both the secondaries
    # that is we are using an optimisitic graph probability equal to 1
    reward_considering_secondary = []
    for primary_id, secondary_choices_list in enumerate(scdy):
        reward_from_primary = 0
        reward_from_primary += margin_values[primary_id] * n_items_bought[primary_id]
        reward_from_primary += np.sum(margin_values[secondary_choices_list] * n_items_bought[secondary_choices_list])
        reward_considering_secondary.append(reward_from_primary)

    # 3. we do another optimistic assumption that is : every user land on the page of the best primary product
    # (the one giving best reward of 2.)
    # 4. multiply by the max possible number of items bought of the user
    n_users = usr.total_users
    alpha_ratios = distribute_alpha([0])[0]
    n_users_on_products = n_users * alpha_ratios
    n_users_on_products = n_users_on_products[1:]

    best_reward = np.sum(reward_considering_secondary * n_users_on_products)

    return best_reward
