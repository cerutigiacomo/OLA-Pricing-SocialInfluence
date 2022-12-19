from resources.Environment import Environment
from step_3.UCBLearner import *

debug = False

def find_clairvoyant_indexes(conv_rates_aggregated):
    prices_of_products = []
    costs_of_products = []
    products = get_product()
    for i in range(len(products)):
        prices_of_products.append(products[i]['price'])
        costs_of_products.append(products[i]['cost'])
    # create an array with the prices(5x4) and the costs(1x5) of every product
    prices_of_products = np.array(prices_of_products)
    costs_of_products = np.array(costs_of_products)

    if debug:
        print("conv_rates_aggregated: \n", conv_rates_aggregated)
        print(prices_of_products)
        print(costs_of_products)

    costs_of_products = costs_of_products.reshape(1, -1)
    margins_of_products = np.subtract(prices_of_products, np.tile(costs_of_products.transpose(), (1, 4)))

    if debug:
        print(margins_of_products)

        # BEST MARGIN FOR A SINGLE UNIT OF PRODUCT
        print(np.multiply(conv_rates_aggregated, margins_of_products))

    best_margin_for_unit = np.argmax(np.multiply(conv_rates_aggregated, margins_of_products), axis=1)
    clairvoyant_margin_values = margins_of_products[np.arange(numbers_of_products), best_margin_for_unit]

    # ???
    best_margin_for_unit = [3, 1, 3, 1, 2]
    print("Indici dei clairvoyant price_index : \n", best_margin_for_unit)
    print("Effettivi valori di margine : \n", clairvoyant_margin_values)
    return best_margin_for_unit, clairvoyant_margin_values


def find_clairvoyant_reward_by_simulation(env):
    sim = env.sim
    users = env.users

    arms = [[_ for x in range(numbers_of_products)] for _ in range(different_value_of_prices)]

    rewards_tot = np.zeros((numbers_of_products, different_value_of_prices))


    N = 300

    for arm in arms:
        sim.prices, sim.margins = get_prices_and_margins(arm)
        sim.prices_index = arm
        for rounds in range(N):
            rewards, *_ = website_simulation(sim,users)
            #if np.sum(rewards) > np.sum(rewards_tot[np.arange(numbers_of_products), arm]):
            for product in range(numbers_of_products):
                if rewards[product] > rewards_tot[product,arm[product]]:
                    rewards_tot[product, arm[product]] = rewards[product]

    #rewards_tot = rewards_tot/N
    best_prices = np.argmax(rewards_tot, axis=1)
    top_revenue = np.sum(rewards_tot[np.arange(numbers_of_products), best_prices])
    return best_prices, top_revenue

def find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation, plot=False):
    # TODO review
    y_clairvoyant = 0
    rew = np.zeros(daily_simulation)
    for i in range(daily_simulation):
        reward, product_visited, items_bought, items_rewards = env.round(clairvoyant_price_index)
        reward = np.sum(reward)
        rew[i] = reward
        y_clairvoyant = ((y_clairvoyant*i) + reward) / (i+1)
    # y_clairvoyant = mean of the reward of the clairvoyant
    y_clairvoyant = np.max(rew)

    if plot:
        x_labels = learner.list_prices
        margin = learner.list_margins

        x_values = [i for i in range(x_labels.shape[0])]

        fig = plt.figure(1, figsize=(70, 12))
        plt.plot(x_values, margin)
        plt.xticks(ticks=x_values, labels=x_labels, rotation=90)
        plt.plot(x_values, np.repeat(y_clairvoyant, len(x_values)), "r-")
        plt.grid()
        plt.show()

    return y_clairvoyant


def find_not_aggregated_reward(best_arm_per_class, env):
    total_reward = 0
    for i in env.users_indexes:
        total_reward += data["users"]["classes"][i]["fraction_between_other_classes"] * \
                        find_reward_per_class(best_arm_per_class[i], i, env)

    # TODO find the best reward for the not aggregated case
    return total_reward

def find_reward_per_class(arm, user_class, env):
    iteration = 20
    rew = np.zeros(iteration)
    environment = Environment(env.n_arms, env.prices, env.prices, env.lam, env.secondary, arm, [user_class], get_users(user_class))
    for i in range(iteration):
        reward, product_visited, items_bought, items_rewards = environment.round(arm)
        reward = np.sum(reward)
        rew[i] = reward
    # y_clairvoyant = mean of the reward of the clairvoyant
    return np.max(rew)
