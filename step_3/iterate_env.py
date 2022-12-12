from clairvoyant import *
from plotting.plot_reward_regret import *
def iterate(learner, env, iteration, daily_simulation, clairvoyant_price_index, name_alg):

    global price_pulled, reward_observed, product_visited, items_bought, items_rewards
    for i in range(iteration):
        for z in range(daily_simulation):
            #learner.debug()
            price_pulled = learner.act()
            reward_observed, product_visited, items_bought, items_rewards = env.round(price_pulled)
            learner.update_pulled_and_success(price_pulled, product_visited, items_bought, items_rewards)
        learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)

    # Plot Regret and Reward
    clairvoyant_margin_values = find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation)
    if debug:
        print("\nReward clairvoyant: ", clairvoyant_margin_values)
    clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin_values)
    cumulative_reward = np.cumsum(learner.list_margins)
    cumulative_regret = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)
    final_reward = learner.list_margins

    plot_regret_reward(cumulative_regret,
                       cumulative_reward,
                       final_reward,
                       clairvoyant_margin_values,
                       label_alg=name_alg,
                       day=iteration)
