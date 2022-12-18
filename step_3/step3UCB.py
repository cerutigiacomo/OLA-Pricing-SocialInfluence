import scipy.special

from Learner.clairvoyant import *
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

learner = UCBLearner(lamb, secondary, [0], 4)
# conversion_rates not observable, then the learner will estimate them.
for i in range(len(class_choosed)):
    learner.users[i].conv_rates = npr.rand(numbers_of_products, different_value_of_prices)

iteration = 150
daily_interaction = 50

#final_reward= np.zeros(iteration)
#cumulative_regret = np.zeros(iteration)
#cumulative_reward = np.zeros(iteration)

env = Environment(different_value_of_prices, prices, margins, lamb, secondary, [0, 0, 0, 0, 0], class_choosed, get_users(class_choosed))
iterate(learner, env, iteration, daily_interaction, clairvoyant_price_index, "step3UCB", 3)


rewards = learner.means
widths = learner.widths

pp = enumerate_price_products(rewards,widths)


fig2 = plt.figure(2,figsize=(30,10))
x_values = np.arange(len(pp))
colors = ["r","g","b","y","m"]
for x in x_values:
    plt.scatter(x,pp[x][2],color=colors[pp[x][0]])
    plt.scatter(x,conv_rates_aggregated.flatten()[x],color="m",marker="x")
    #plt.vlines(x=x,ymin=pp[x][2]-pp[x][3],ymax=pp[x][2]+pp[x][3],colors=colors[pp[x][0]])
    plt.errorbar(x=x,y=pp[x][2],yerr=pp[x][3],color=colors[pp[x][0]],capsize=3)
plt.xticks(x_values)
plt.ylim(-2, 3)
plt.grid()
plt.show()


# TODO : add as a function if needed

def _upper_bound_regret_TS():
    clairvoyant_margin_values = find_clairvoyant_reward(learner, env, clairvoyant_price_index, 30)
    clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin_values)
    observed_reward = learner.list_margins
    delta_reward = np.subtract(clairvoyant_margin_iterated, observed_reward)

    kl = scipy.special.kl_div(clairvoyant_margin_iterated,observed_reward)

    c = np.log(learner.t) + np.log(np.log(learner.t))
    upper_bound_regret_list = []
    for val,kl in zip(delta_reward,kl):
        if val > 0:
            x = (val * c / kl)
            upper_bound_regret_list.append(x)

    eps = 0.001
    upper_bound_regret = (1+eps)*np.sum(upper_bound_regret_list)
    pass