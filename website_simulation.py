import numpy.random as npr
from resources.define_distribution import *
from plotting.plot_distributions import *
import json

f = open('../resources/environment.json')
data = json.load(f)
numbers_of_products = data["product"]["numbers_of_products"]
days_simulation = data["simulator"]["days"]


def website_simulation(sim, users):
    # This method simulates users visiting the ecommerce website
    # returns total rewards for all five products
    total_rewards = np.zeros(5, np.float16)
    for user_class in users:
        total_rewards += run_for_alpha_ratio(user_class, sim)

    return total_rewards


def simulate_multiple_days(sim, users, classes_idx, days=days_simulation):
    total_reward = np.zeros(numbers_of_products)
    for j in range(days):
        reward = website_simulation(sim, users)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print("total revenue ", reward)
        total_reward += reward
        # Change the prices, alpha, total_users daily
        new_alpha = distribute_alpha(classes_idx)
        new_total_users = distribute_total_user(classes_idx)
        for i in classes_idx:
            users[i].alpha = new_alpha[i]
            users[i].total_users = new_total_users[i]
        sim.prices_greedy, sim.margins, sim.today = distribute_prices()

    # plot_reward(total_reward)
    return total_reward


""" From the text
"In practice, you can only consider the 𝛼 ratios and disregard the total number of users."
We use the Alpha ratio and the total number of users. for iterating on the single product
"""


def run_for_alpha_ratio(user_class, sim):
    rewards_count = []
    visited = []
    total_rewards = np.zeros(5, np.float16)
    alpha = user_class.alpha
    for i in range(numbers_of_products + 1):
        # i = 0 is the competitor
        if i == 0:
            continue
        for j in range(int(alpha[i] * user_class.total_users)):
            sim.reset()
            rewards = sim.simulation(i - 1, user_class)
            visited.append(sim.visited_primaries)
            rewards_count.append(np.sum(rewards))
            total_rewards += rewards

    # plot_result_simulation(rewards_count, visited)
    return total_rewards
