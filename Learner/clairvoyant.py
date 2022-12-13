from step_3.UCBLearner import *

debug = False

def find_clairvoyant_indexes(conv_rates_aggregated):
    prices_of_products = []
    costs_of_products = []
    products = get_product()
    for i in range(len(products)):
        prices_of_products.append(products[i]['price'])
        costs_of_products.append(products[i]['cost'])
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

    best_margin_for_unit = np.argmax(margins_of_products, axis=1)
    clairvoyant_margin_values = margins_of_products[np.arange(numbers_of_products), best_margin_for_unit]

    best_margin_for_unit = [3, 1, 3, 1, 2]
    print("Indici dei clairvoyant price_index : \n", best_margin_for_unit)
    print("Effettivi valori di margine : \n", clairvoyant_margin_values)
    return best_margin_for_unit, clairvoyant_margin_values


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
