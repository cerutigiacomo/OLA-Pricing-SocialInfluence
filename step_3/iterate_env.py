from clairvoyant import *
from plotting.plot_reward_regret import *
def iterate(learner, env, iteration, claivoyant_price_index, name_alg, n_step):

    for iterations in range(iteration):
        learner.debug()
        price_pulled = learner.act()
        reward_observed, product_visited, items_bought, items_rewards = env.round(price_pulled)
        learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)
        learner.update_step_parameters(product_visited, items_bought, n_step)

    # Plot Regret and Reward
    clairvoyant_margin = find_clairvoyant_reward(learner,env,claivoyant_price_index,iteration)
    clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin)
    cumulative_reward = np.cumsum(learner.list_margins)
    cumulative_regret = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)
    final_reward = learner.list_margins

    plot_regret_reward(cumulative_regret,
                       cumulative_reward,
                       final_reward,
                       clairvoyant_margin,
                       label_alg=name_alg,
                       day=iteration)
