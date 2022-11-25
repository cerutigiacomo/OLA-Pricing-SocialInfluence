import numpy as np
import numpy.random as npr
import json


f = open('../resources/environment.json')
data = json.load(f)
max_margin = data["product"]["max_margin"]
different_value_of_prices = data["product"]["different_value_of_prices"]
numbers_of_products = data["product"]["numbers_of_products"]
classes = data["users"]["classes"]
users_classes = len(classes)


def simulator_distribution():
    # For every product, order the prices in increasing levels. Every price is associated with a known margin.
    prices, margins, today = distribute_prices()

    # Define the pair and the order of the secondary products to display
    secondary = np.zeros((numbers_of_products, 2))
    all_contained = False
    # Every product should have different secondary product (different from himself)
    # Every product has to be reachable, so all the product has to be contained in the secondary array [????]
    while not all_contained:
        for i in range(numbers_of_products):
            secondary[i] = npr.choice(numbers_of_products, 2, replace=False)
            while np.isin(secondary[i], i).any():
                secondary[i] = npr.choice(numbers_of_products, 2, replace=False)
        values = np.isin(secondary, list(range(numbers_of_products)))
        all_contained = values.all()

    lamb = data["product"]["lambda"]  # LAMBDA

    # EXPLANATION OF THE DISTRIBUTIONS
    """
        Firstly we set 3 constant
            - max_margin: the maximum value of the margin value (not sure)
            - different_value_of_prices: we have 4 values of prices for every product
            - numbers_of_products: the website sell 5 different products
    
        Margin: type matrix [4,5], were on the 1st dimension we have the different prices per product
                and on the 2nd dimension we have the different product
                Thn we random select [maybe normally or with another type of distribution can be better]
                the margin value considering the max_margin constant
        Prices: type array [5,1], were we have stored the index of the selected margin for the day
            i think that a random distribution for choosing which between the already generated margin to use is correct.
    """
    return prices, margins, secondary, today

def distribute_prices():
    today = int(npr.choice(different_value_of_prices))
    products = data["product"]["products"]

    prices = [products[i]["price"][today] for i in range(numbers_of_products)]
    margin = [products[i]["price"][today]-products[i]["cost"] for i in range(numbers_of_products)]
    return prices, margin, today


def distribute_alpha():
    alpha_ratios = np.zeros((users_classes,numbers_of_products+1))
    for i in range(users_classes):
        product_weight = classes[i]["alpha"]["alpha_weights"]
        alpha_ratios[i] = npr.dirichlet(product_weight, 1).reshape(numbers_of_products+1)
    # 6 ratios, the first is for the COMPETITOR webpage
    """
    distribution option:  -   dirichlet (forced)
    distribution weights are chosen random.
    """
    return alpha_ratios

def distribute_total_user():
    total_users = np.zeros(users_classes)

    for i in range(users_classes):
        mean_users_per_class = classes[i]["total_user"]["mean_users_per_class"]
        user_per_class_std = classes[i]["total_user"]["user_per_class_std"]
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

def user_distribution():

    #                                           1 Create the demand curves (conversion rates)
    conv_rates = np.zeros((users_classes, different_value_of_prices, numbers_of_products))
    # TODO generate conv factor for every prices of every product
    for i in range(users_classes):
        min_demand = classes[i]["demand"]["min_demand"]
        max_demand = classes[i]["demand"]["max_demand"]
        # Define a maximum value of conversion rates for every user class
        conv_rates[i] = npr.uniform(min_demand, max_demand, (different_value_of_prices, numbers_of_products))

    """
        Conversion rates are chosen uniform between 0 and a max[i]
        max_demand[i] is the maximum value of conversion rate of a single user class, random chosen
    """

    #                                           2 Create num of users
    total_users = distribute_total_user()

    #                                           3 Create alpha ratios
    alpha_ratios = distribute_alpha()

    #                                           4 Create number of products sold
    # Uncorrelated with the actual price ??
    n_items_bought = np.zeros((users_classes, numbers_of_products))
    for i in range(users_classes):
        max_item_bought = classes[i]["n_items_buyed"]["max_item_bought"]
        n_items_bought[i] = npr.uniform(0, max_item_bought, numbers_of_products).astype(int)

    """
    From the text: 
        the number of items a user will buy is a random variable independent of any other variable;
    """

    #                                           5 Create graph probabilities
    # FULLY CONNECTED
    graph_weights = np.zeros((users_classes, numbers_of_products, numbers_of_products))
    max_demand = np.zeros(users_classes)

    for i in range(users_classes):
        min_graph_probability = classes[i]["graph"]["min_probability"]
        max_graph_probability = classes[i]["graph"]["max_probability"]
        # Define a maximum value of graph_weight for every user class
        max_demand[i] = npr.uniform(min_graph_probability, max_graph_probability)
        graph_weights[i] = npr.uniform(0, max_demand[i], (numbers_of_products, numbers_of_products))

    # second type of graph set to 0 only certain values
    # NOT FULLY CONNECTED
    graph_weights_not_fully_connected = np.copy(graph_weights)
    for i in range(numbers_of_products):
        how_many_set_to_zero = int(round(numbers_of_products * npr.random()))
        set_to_zero = (npr.choice(numbers_of_products, how_many_set_to_zero, replace=False)).astype(int)

        for j in set_to_zero:
            j = int(j)
            graph_weights_not_fully_connected[:, i, j] = 0

    if data["users"]["graph_weight_fully_connected"]:
        graph = graph_weights
    else:
        graph = graph_weights_not_fully_connected

    return total_users, alpha_ratios, graph, n_items_bought, conv_rates