import numpy as np
import numpy.random as npr
import json
from users  import Users_group

f = open('../resources/environment.json')
data = json.load(f)
different_value_of_prices = data["product"]["different_value_of_prices"]
numbers_of_products = data["product"]["numbers_of_products"]
classes = data["users"]["classes"]
users_classes = len(classes)
npr.seed(data["simulator"]["seed"])


def simulator_distribution():
    # For every product, order the prices in increasing levels. Every price is associated with a known margin.
    prices, margins, today = distribute_prices()

    # Define the pair and the order of the secondary products to display
    secondary = data["simulator"]["secondary"]

    return prices, margins, secondary, today


def distribute_prices():
    today = int(npr.choice(different_value_of_prices))
    products = get_product()

    prices = [products[i]["price"][today] for i in range(numbers_of_products)]
    margin = [products[i]["price"][today] - products[i]["cost"] for i in range(numbers_of_products)]
    return prices, margin, today


def get_product():
    products = data["product"]["products"]
    return products


def distribute_alpha(classes_idx):
    alpha_ratios = np.zeros((len(classes_idx), numbers_of_products + 1))
    for i in range(len(classes_idx)):
        product_weight = classes[classes_idx[i]]["alpha"]["alpha_weights"]
        alpha_ratios[i] = npr.dirichlet(product_weight, 1).reshape(numbers_of_products + 1)
    # 6 ratios, the first is for the COMPETITOR webpage
    """
    distribution option:  -   dirichlet (forced)
    distribution weights are chosen random.
    """
    return alpha_ratios


def distribute_total_user(classes_idx):
    total_users = np.zeros(len(classes_idx))

    for i in range(len(classes_idx)):
        mean_users_per_class = classes[classes_idx[i]]["total_user"]["mean_users_per_class"]
        user_per_class_std = classes[classes_idx[i]]["total_user"]["user_per_class_std"]
        total_users[i] = (npr.normal(mean_users_per_class,
                                     user_per_class_std, 1)).astype(int)

    # ::int array randomly from 0 to max_users_per_class
    """
    distribution option: 
    -   Normal [i think can be a correct idea like implemented]
    -   Random 
    -   ....
    the mean is a random variable from 0 to 300
    the std is a random variable from 0 to 30
    """
    return total_users


def user_distribution(classes_idx=None):
    if classes_idx is None:
        classes_idx = [0]

    features = []
    for i in range(len(classes_idx)):
        features.append(classes[classes_idx[i]]["features"])

    #                                           1 Create the demand curves (conversion rates)
    conv_rates = np.zeros((len(classes_idx), numbers_of_products, different_value_of_prices))
    for i in range(len(classes_idx)):
        min_demand = classes[classes_idx[i]]["demand"]["min_demand"]
        max_demand = classes[classes_idx[i]]["demand"]["max_demand"]
        # Define a maximum value of conversion rates for every user class
        conv_rates[i] = npr.uniform(min_demand, max_demand, (numbers_of_products, different_value_of_prices))

    """
        Conversion rates are chosen uniform between 0 and a max[i]
        max_demand[i] is the maximum value of conversion rate of a single user class, random chosen
    """

    #                                           2 Create num of users
    total_users = distribute_total_user(classes_idx)

    #                                           3 Create alpha ratios
    alpha_ratios = distribute_alpha(classes_idx)

    #                                           4 Create number of products sold
    # Uncorrelated with the actual price ??
    n_items_bought = np.zeros((len(classes_idx), numbers_of_products))
    for i in range(len(classes_idx)):
        max_item_bought = classes[classes_idx[i]]["n_items_buyed"]["max_item_bought"]
        n_items_bought[i] = npr.uniform(1, max_item_bought, numbers_of_products).astype(int)

    """
    From the text: 
        the number of items a user will buy is a random variable independent of any other variable;
    """

    #                                           5 Create graph probabilities
    # FULLY CONNECTED
    graph_weights = np.zeros((len(classes_idx), numbers_of_products, numbers_of_products))
    max_demand = np.zeros(len(classes_idx))

    for i in range(len(classes_idx)):
        min_graph_probability = classes[classes_idx[i]]["graph"]["min_probability"]
        max_graph_probability = classes[classes_idx[i]]["graph"]["max_probability"]
        # Define a maximum value of graph_weight for every user class
        max_demand[i] = npr.uniform(min_graph_probability, max_graph_probability)
        graph_weights[i] = npr.uniform(0, max_demand[i], (numbers_of_products, numbers_of_products))
        for j in range(numbers_of_products):
            graph_weights[i][j][j] = 0
    # second type of graph set to 0 only certain values
    # NOT FULLY CONNECTED
    graph_weights_not_fully_connected = np.copy(graph_weights)
    for z in range(len(classes_idx)):
        for i in range(numbers_of_products):
            how_many_set_to_zero = int(round((numbers_of_products-2) * npr.random()))
            set_to_zero = (npr.choice(numbers_of_products, how_many_set_to_zero, replace=False)).astype(int)

            for j in set_to_zero:
                j = int(j)
                graph_weights_not_fully_connected[z, i, j] = 0

    if data["users"]["graph_weight_fully_connected"]:
        graph = graph_weights
    else:
        graph = graph_weights_not_fully_connected

    return total_users, alpha_ratios, graph, n_items_bought, conv_rates, features

def get_users(classes_idx=None):
    if classes_idx is None:
        classes_idx = [0]
    users = []
    total_users, alpha_ratios, graph, n_items_bought, conv_rates, features = user_distribution(classes_idx)
    for i in range(len(classes_idx)):
        users.append(Users_group(total_users, alpha_ratios[i], graph[i], n_items_bought[i], conv_rates[i], features[i]))

    return users


def get_prices_and_margins(index):
    products = get_product()
    prices = [products[i]["price"][index[i]] for i in range(numbers_of_products)]
    margin = [products[i]["price"][index[i]] - products[i]["cost"] for i in range(numbers_of_products)]
    return prices, margin