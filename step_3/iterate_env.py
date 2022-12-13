from Learner.clairvoyant import *
from plotting.plot_reward_regret import *

# 5 mean as the possible product to buy
mean_iteration = 5
def iterate(learner, env, iteration, daily_simulation, clairvoyant_price_index, name_alg, n_step):

    global price_pulled, reward_observed, product_visited, items_bought, items_rewards, clairvoyant_margin_values

    cumulative_reward = np.zeros((mean_iteration, iteration))
    cumulative_regret = np.zeros((mean_iteration, iteration))
    final_reward = np.zeros((mean_iteration, iteration))

    for z in range(mean_iteration):
        print("plot ite: ", z)
        # re-initialized lambda, secondary prod, user classes, no of arms
        learner.reset()

        for i in range(iteration):
            learner.debug()
            price_pulled = learner.act()
            reward_observed, product_visited, items_bought, items_rewards = env.round(price_pulled)
            learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)
            #learner.update_pulled_and_success(price_pulled, product_visited, items_bought, items_rewards)

        """
        ucb
        
        
                        learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)
                        learner.update_step_parameters(product_visited, items_bought, n_step)
                        
        prima del mio update
        
        """


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
