from Learner.clairvoyant import *
from plotting.plot_reward_regret import *


mean_iteration = 2
def iterate(user_conv_rates, changes_instant, learner, env, iteration, daily_simulation, clairvoyant_price_index, name_alg, n_step=3):

    global price_pulled, reward_observed, product_visited, items_bought, items_rewards, clairvoyant_margin_values

    cumulative_reward = np.zeros((mean_iteration,iteration))
    cumulative_regret = np.zeros((mean_iteration,iteration))
    final_reward = np.zeros((mean_iteration,iteration))
    product_visited_list = []
    items_bought_list = []
    cumulative_regret_second = np.zeros((mean_iteration, int(iteration / 2)))
    cumulative_regret_first = np.zeros((mean_iteration, int(iteration / 2)))

    for z in range(mean_iteration):
        product_visited_list = []
        items_bought_list = []
        print("plot ite: ", z)
        # re-initialized lambda, secondary prod, user classes, no of arms
        learner.reset()
        env.users[0].conv_rates = user_conv_rates
        learner.users[0].conv_rates = user_conv_rates
        print("ACTUAL CONV_RATES:\n", user_conv_rates)

        for i in range(iteration):

            learner.users[0].conv_rates = env.users[0].conv_rates
            if i == env.changes_instant[0]:
                # Random conversion rate (demand curve) generating number between [0, 1)
                # one matrix simulating a class of users subjected to abrupt changes
                changed_conv_rates = np.array([[0.143, 0.125, 0.765, 0.999],
                                               [0.123, 0.224, 0.934, 0.234],
                                               [0.234, 0.987, 0.593, 0.077],
                                               [0.221, 0.876, 0.265, 0.123],
                                               [0.721, 0.234, 0.013, 0.376]])
                env.users[0].conv_rates = changed_conv_rates
                learner.users[0].conv_rates = changed_conv_rates
                print(" NEW CONVERSION RATE ARE: \n", changed_conv_rates)

            print("iteration: ", i)
            learner.debug()
            price_pulled = learner.act()
            reward_observed, product_visited, items_bought, items_rewards = env.round_6(price_pulled)
            product_visited_list += product_visited[0]
            items_bought_list += items_bought[0]
            learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)

            if i < env.changes_instant[0]:
                clairvoyant_margin_values = find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation)
                print("CLAIRVOYANT", clairvoyant_margin_values)
            else:
                clairvoyant_margin_values_new = find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation)
                print("CLAIRVOYANT AFTER ABRUPT CHANGES", clairvoyant_margin_values_new)

            if i == env.changes_instant[0] - 1:
                clairvoyant_margin_iterated = np.full(int(iteration/2), clairvoyant_margin_values)
                cumulative_regret_first[z, :] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)

            if i == (env.changes_instant[0] * 2) - 1:
                clairvoyant_margin_iterated = np.full(int(iteration/2), clairvoyant_margin_values_new)
                # to obtain the second half of list_margins
                x = learner.list_margins[len(learner.list_margins)//2:]
                cumulative_regret_second[z, :] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(x)

        clairvoyant_margin_iterated = np.zeros(iteration, clairvoyant_margin_values)
        clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin_values)
        cumulative_reward[z,:] = np.cumsum(learner.list_margins)
        cumulative_regret[z,:] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)
        final_reward[z,:] = learner.list_margins

    #SLIDING WINDOW ITERATION
    cumulative_rewardSW = np.zeros((mean_iteration,iteration))
    cumulative_regretSW = np.zeros((mean_iteration,iteration))
    final_rewardSW = np.zeros((mean_iteration,iteration))
    product_visited_list = []
    items_bought_list = []
    cumulative_regret_second = np.zeros((mean_iteration, int(iteration / 2)))
    cumulative_regret_first = np.zeros((mean_iteration, int(iteration / 2)))

    for z in range(mean_iteration):
        product_visited_list = []
        items_bought_list = []
        print("plot ite: ", z)
        # re-initialized lambda, secondary prod, user classes, no of arms
        learner.reset()
        env.users[0].conv_rates = user_conv_rates
        learner.users[0].conv_rates = user_conv_rates
        print("ACTUAL CONV_RATES:\n", user_conv_rates)

        for i in range(iteration):

            learner.users[0].conv_rates = env.users[0].conv_rates
            if i == env.changes_instant[0]:
                # Random conversion rate (demand curve) generating number between [0, 1)
                # one matrix simulating a class of users subjected to abrupt changes
                changed_conv_rates = np.array([[0.143, 0.125, 0.765, 0.999],
                                               [0.123, 0.224, 0.934, 0.234],
                                               [0.234, 0.987, 0.593, 0.077],
                                               [0.221, 0.876, 0.265, 0.123],
                                               [0.721, 0.234, 0.013, 0.376]])
                env.users[0].conv_rates = changed_conv_rates
                learner.users[0].conv_rates = changed_conv_rates
                print(" NEW CONVERSION RATE ARE: \n", changed_conv_rates)

            print("iteration: ", i)
            learner.debug()
            price_pulled = learner.act()
            reward_observed, product_visited, items_bought, items_rewards = env.round_6(price_pulled)
            product_visited_list += product_visited[0]
            items_bought_list += items_bought[0]
            learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)

            if i < env.changes_instant[0]:
                clairvoyant_margin_values = find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation)
                print("CLAIRVOYANT", clairvoyant_margin_values)
            else:
                clairvoyant_margin_values_new = find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation)
                print("CLAIRVOYANT AFTER ABRUPT CHANGES", clairvoyant_margin_values_new)

            if i == env.changes_instant[0] - 1:
                clairvoyant_margin_iterated = np.full(int(iteration/2), clairvoyant_margin_values)
                cumulative_regret_first[z, :] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)

            if i == (env.changes_instant[0] * 2) - 1:
                clairvoyant_margin_iterated = np.full(int(iteration/2), clairvoyant_margin_values_new)
                # to obtain the second half of list_margins
                x = learner.list_margins[len(learner.list_margins)//2:]
                cumulative_regret_second[z, :] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(x)

        clairvoyant_margin_iterated = np.zeros(iteration, clairvoyant_margin_values)
        clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin_values)
        cumulative_rewardSW[z,:] = np.cumsum(learner.list_margins)
        cumulative_regretSW[z,:] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)
        final_rewardSW[z,:] = learner.list_margins


    # Plot Regret and Reward
    plot_regret_reward_UCB6(cumulative_regret,
                            cumulative_regretSW,
                             cumulative_regret_first,
                             cumulative_regret_second,
                             cumulative_reward,
                             cumulative_rewardSW,
                             final_reward,
                            final_rewardSW,
                             clairvoyant_margin_values,
                             clairvoyant_margin_values_new,
                             label_alg=name_alg,
                             day=iteration)