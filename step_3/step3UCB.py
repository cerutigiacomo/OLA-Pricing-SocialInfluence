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


######### UCB
# TODO iterate the learner more times and get the mean of the results

learner = UCBLearner(lamb, secondary, users, 4, [0])

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
y_clairvoyant = find_clairvoyant_reward(learner, clairvoyant_price_index)

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
    max_value = np.max(means.flatten())
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

rewards = scale_min_max(learner.means)
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

