import numpy as np

def website_simulation(sim, user_class):
    # This method simulates users visiting the ecommerce website
    # argument is an User class instance
    # returns total rewards for all five products

    total_rewards = np.zeros(5, np.float16)

    for j in range(len(user_class.n_users) - 1):
        # Reward of the single product
        product_reward = np.zeros(5, np.float16)
        for n in range(round(user_class.n_users[j + 1])):
            sim.visited_primaries = []
            rewards = sim.simulation(j, user_class)
            product_reward += rewards

        total_rewards += product_reward
        # print(round(user_class.n_users[j+1]),"users landing on product", j+1 ,product_reward)

    return total_rewards
