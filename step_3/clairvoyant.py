from UCBLearner import *

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

    best_margin_for_unit = np.argmax(np.multiply(conv_rates_aggregated, margins_of_products), axis=1)
    clairvoyant_margin_values = margins_of_products[np.arange(numbers_of_products), best_margin_for_unit]

    print("Indici dei clairvoyant price_index : \n", best_margin_for_unit)
    print("Effettivi valori di margine : \n", clairvoyant_margin_values)
    return best_margin_for_unit, clairvoyant_margin_values


def find_clairvoyant_reward(learner, clairvoyant_price_index):
    x_labels = learner.list_prices
    margin = learner.list_margins

    x_values = [i for i in range(x_labels.shape[0])]

    # TODO : clairvoyant solution
    # a first attempt and reported below is to simply simulate with the best price setting of clairvoyant
    y_clairvoyant = np.sum(learner.simulate(clairvoyant_price_index))
    # a second attempt could be to use a different simulator (for example by setting it to the higher possbile n_boughts)
    # but i have doubt about how to handle the graph secondaries, and alpha ratios, ...
    # NOTE : running the algorithm for 1000 or more iterations shows the limit of this clairvoyant solution

    fig = plt.figure(1, figsize=(70, 12))
    plt.plot(x_labels, margin)
    plt.xticks(ticks=x_values, labels=x_labels, rotation=90)
    plt.plot(x_values, np.repeat(y_clairvoyant, len(x_values)), "r-")
    plt.grid()
    plt.show()

    return y_clairvoyant
