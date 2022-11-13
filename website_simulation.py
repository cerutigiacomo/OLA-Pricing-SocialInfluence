import numpy as np
import numpy.random as npr
number_of_product = 5


def website_simulation(sim, user_class):
    # This method simulates users visiting the ecommerce website
    # argument is an User class instance
    # returns total rewards for all five products

    total_rewards = np.zeros(5, np.float16)

    for i in range(user_class.total_users - 1):
        # Reward of the single product
        # product_reward = np.zeros(5, np.float16)
        # for n in range(round(user_class.total_users)):
        sim.visited_primaries = []
        # TODO select the correct product by using alpha probabilities
        j = int(np.random.choice(number_of_product+1, 1, p=user_class.alpha))
        if j == 0:
            # The competitor has been selected!
            continue
        j -= 1
        rewards = sim.simulation(j, user_class)
        total_rewards += rewards

        # total_rewards += product_reward
        # print(round(user_class.total_users[j+1]),"users landing on product", j+1 ,product_reward)

    return total_rewards
