from clairvoyant import *
from plotting.plot_reward_regret import *
def iterate(learner, env, iteration, clairvoyant_margin_values, name_alg):

    for iterations in range(iteration):
        learner.debug()
        price_pulled = learner.act()
        reward_observed, product_visited, items_bought, items_rewards = env.round(price_pulled)
        learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)

    # Plot Regret and Reward
    clairvoyant_margin = clairvoyant_margin_values
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
