from plotting.plot_reward_regret import *
from clairvoyant import *

f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]
debug = False


prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
# TODO : update aggregate data which has been taken from Student for test purposes
users = get_users([0])

# TODO : again i have the Student for test purposes
conv_rates_aggregated = users[0].conv_rates
clairvoyant_price_index, clairvoyant_margin_values = find_clairvoyant_indexes(conv_rates_aggregated)


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

    best_reward = np.sum(reward_considering_secondary*n_users_on_products)

    return best_reward


users = get_users([0])
# TODO : using Student user for test purposes
user = users[0]
# used below and passed as parameter to ucblearner
max_reward = find_max_rewards(user,secondary)

# TODO : WORKING TEST does not works properly
# è troppo top reward per una simulazione aleatoria reale
max_reward = max_reward/10




######### UCB
# TODO iterate the learner more times and get the mean of the results

learner = UCBLearner(lamb, secondary, users, 4, max_reward)

iteration = 1000

final_reward= np.zeros(iteration)
cumulative_regret = np.zeros(iteration)
cumulative_reward = np.zeros(iteration)

for iterations in range(iteration):
    learner.debug()
    price_pulled = learner.act()
    reward_observed = learner.simulate(price_pulled)
    learner.update(price_pulled, reward_observed)

# Clairvoyant solution
y_clairvoyant = find_clairvoyant_reward(learner, clairvoyant_price_index, iteration)

# Plot UCB Regret and Reward
clairvoyant_margin = y_clairvoyant
clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin)
cumulative_reward = np.cumsum(learner.list_margins)
cumulative_regret = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)
final_reward = learner.list_margins

plot_regret_reward(cumulative_regret,
                   cumulative_reward,
                   final_reward,
                   clairvoyant_margin,
                   label_alg= "Step3UCB",
                   day = iteration)


# show arms confidence
def scale_min_max(means):
    #max_value = np.max(means.flatten())
    max_value = max_reward
    min_value = np.min(means.flatten())
    x_scaled = (means - min_value) / (max_value - min_value)
    return x_scaled

def enumerate_price_products(rewards,widths):
    pp = []
    # TODO : scaled versions is an attempt solution for the expected reward problem
    # trple of values (product,price,expected_reward_SCALED,confidence)
    for i in range(0,numbers_of_products):
        for j in range(0,learner.n_arms):
            pp.append((i,j,rewards[i,j],widths[i,j]))
    return pp


#rewards = scale_min_max(learner.means)
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

