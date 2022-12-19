from Environment import Environment
from resources.define_distribution import *
import numpy as np
import matplotlib.pyplot as plt

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

    best_margin_for_unit = np.argmax(margins_of_products, axis=1)
    clairvoyant_margin_values = margins_of_products[np.arange(numbers_of_products), best_margin_for_unit]

    best_margin_for_unit = [3, 1, 3, 1, 2]
    print("Indici dei clairvoyant price_index : \n", best_margin_for_unit)
    print("Effettivi valori di margine : \n", clairvoyant_margin_values)
    return best_margin_for_unit, clairvoyant_margin_values


def find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation):
    rew = np.zeros(daily_simulation)
    for i in range(daily_simulation):
        reward, product_visited, items_bought, items_rewards = env.round(clairvoyant_price_index)
        reward = np.sum(reward)
        rew[i] = reward
    return np.max(rew)


def find_not_aggregated_reward(best_arm_per_class, env):
    total_reward = 0
    tot_2 = np.zeros(len(env.users_indexes))
    for i in env.users_indexes:
        tot_2[i] += find_reward_per_class(best_arm_per_class[i], i, env)
        total_reward += tot_2[i] # data["users"]["classes"][i]["fraction_between_other_classes"] * tot_2[i]

    # TODO find the best reward for the not aggregated case
    return total_reward


def find_reward_per_class(arm, user_class, env):
    iteration = env.daily_iteration_mean
    rew = np.zeros(iteration)
    # in this env we only iterate 1 time for not loosing the max reward on the mean
    environment = Environment(env.n_arms, env.prices, env.prices, arm, [user_class], 0)
    for i in range(iteration):
        reward, product_visited, items_bought, items_rewards = environment.round(arm)
        reward = np.sum(reward)
        rew[i] = reward
    # y_clairvoyant = mean of the reward of the clairvoyant
    return np.max(rew)
