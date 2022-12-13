from step_3.clairvoyant import *
from plotting.plot_reward_regret import *

# 5 mean as the possible product to buy
mean_iteration = 5


def iterate(learner, env, iteration, daily_simulation, clairvoyant_price_index, name_alg, n_step):
    global price_pulled, reward_observed, product_visited, items_bought, items_rewards, clairvoyant_margin_values

    cumulative_reward = np.zeros((mean_iteration, iteration))
    cumulative_regret = np.zeros((mean_iteration, iteration))
    final_reward = np.zeros((mean_iteration, iteration))

    # Random conversion rate (demand curve) generating number between [0, 1)
    # one matrix simulating a class of users subjected to abrupt changes
    changed_conv_rates = np.array([[0.943, 0.125, 0.765, 0.999],
                                   [0.123, 0.224, 0.934, 0.234],
                                   [0.234, 0.987, 0.893, 0.677],
                                   [0.321, 0.876, 0.765, 0.123],
                                   [0.321, 0.234, 0.013, 0.876]])

    #print("ECCO I CONV RATESSSSSS", learner.users[0].conv_rates)

    for z in range(mean_iteration):
        print("plot iteration:", z, "-----------------------------------")
        learner.reset()

        for i in range(iteration):

            if i == 50:
                print("CAMBIO---------------------------------------------")
                # TODO: check the variables to reset for the abrupt change
                learner.users[0].conv_rates = changed_conv_rates
                cumulative_reward = np.zeros((mean_iteration, iteration))
                cumulative_regret = np.zeros((mean_iteration, iteration))
                final_reward = np.zeros((mean_iteration, iteration))
                env.sim.items_bought = np.zeros(5)
                env.sim.items_reward = np.zeros(5)
                learner.arm_counters = np.zeros((5,4))

            learner.debug()
            price_pulled = learner.act()
            reward_observed, product_visited, items_bought, items_rewards = env.round(price_pulled)
            # SIMULATE
            learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)
            #learner.update_pulled_and_success(price_pulled, product_visited, items_bought, items_rewards)

        clairvoyant_margin_values = find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation)

        if debug:
            print("\n Reward clairvoyant: ", clairvoyant_margin_values)
        # Return a new array of given shape (iteration), filled with clairvoyant_margin_values
        clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin_values)
        cumulative_reward[z, :] = np.cumsum(learner.list_margins)
        cumulative_regret[z, :] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)
        final_reward[z, :] = learner.list_margins

    # Plot Regret and Reward
    plot_regret_reward(cumulative_regret,
                       cumulative_reward,
                       final_reward,
                       clairvoyant_margin_values,
                       label_alg=name_alg,
                       day=iteration)


