import numpy as np

from  resources.define_distribution import *

def compute_sample_conv_rate(product_visited, items_bought):
    n_user_classes = len(product_visited)

    sample_conv_rates = np.zeros(shape=numbers_of_products)
    counters = np.zeros(shape=numbers_of_products)
    conv_rate_list = []
    for i in range(n_user_classes):
        # product_visited[i] to be accessed which is of length user for that class
        # TODO : cumulative daily counters of seen and bought to estimate a sort of cumulative day-wise conv rate estimate
        # i.e. the mean value of conv rate is the mean of each possible observation
        # of seen and bought from the first day of website observation

        for visited_usr_j, bought_usr_j in zip(product_visited[i], items_bought[i]):
            seen = np.zeros(shape=numbers_of_products)
            seen[visited_usr_j] += 1
            bought = np.zeros(shape=numbers_of_products)
            bought[bought_usr_j > 0.0] += 1

            counters += seen
            mask_seen = seen > 0

            conv_rate_usr_j = np.zeros(shape=numbers_of_products)
            for t in range(numbers_of_products):
                if mask_seen[t]:
                    conv_rate_usr_j[t] = bought[t]/seen[t]
            conv_rate_list.append(conv_rate_usr_j)


    conv_rates = np.array(conv_rate_list)
    conv_rates = np.reshape(conv_rates, newshape=(-1, numbers_of_products))

    for i in range(5):
        if counters[i] == 0:
            conv_rates[i] = 0
            counters[i] = 1

    sample_conv_rates = np.divide(np.sum(conv_rates, axis=0), counters)

    for i in range(sample_conv_rates.shape[0]):
        if np.isnan(sample_conv_rates[i]).any():
            print("NAN\n", np.isnan(sample_conv_rates), "\n sample_conv_rates:\n", sample_conv_rates)
            print("counters:\n", counters, "\n np.sum(conv_rate_list): \n", np.sum(conv_rates))

    return sample_conv_rates


def compute_sample_n_bought(items_visiteed, items_bought):
    boughts_of_product = np.zeros(shape=numbers_of_products)
    seen = np.zeros(shape=numbers_of_products)

    for visited_by_class, bought_by_class in zip(items_visiteed, items_bought):
        for visited_usr, bought_usr in zip(visited_by_class, bought_by_class):
            seen[visited_usr] += 1
            boughts_of_product += bought_usr

    mean_bought = boughts_of_product / seen
    #mean_bought = np.where(~np.isnan(mean_bought), mean_bought, 0)

    return mean_bought





def compute_sample_alpha_ratios():
    pass


def compute_sample_graph_weights():
    pass
