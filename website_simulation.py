import numpy as np
import numpy.random as npr
import json

f = open('../resources/environment.json')
data = json.load(f)
numbers_of_products = data["product"]["numbers_of_products"]

def website_simulation(sim, user_class):
    # This method simulates users visiting the ecommerce website
    # argument is an User class instance
    # returns total rewards for all five products

    total_rewards = np.zeros(5, np.float16)

    # return run_for_total_user(user_class, sim, total_rewards)
    return run_for_alpha_ratio(user_class, sim, total_rewards)


def run_for_total_user(user_class, sim, total_rewards):
    for i in range(user_class.total_users - 1):
        sim.visited_primaries = []
        j = int(np.random.choice(numbers_of_products+1, 1, p=user_class.alpha))
        if j == 0:
            # The competitor has been selected!
            continue
        j -= 1
        rewards = sim.simulation(j, user_class)
        total_rewards += rewards

        return total_rewards

""" From the text

"In practice, you can only consider the 𝛼 ratios and disregard the total number of users."
We use the Alpha ratio and the total number of users. for iterating on the single product
"""
def run_for_alpha_ratio(user_class, sim, total_rewards):
    alpha = user_class.alpha
    for i in range(numbers_of_products+1):
        # i = 0 is the competitor
        if i == 0:
            continue
        for j in range(int(alpha[i] * user_class.total_users)):
            rewards = sim.simulation(i-1, user_class)
            total_rewards += rewards

    return total_rewards