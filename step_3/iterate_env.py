import copy

from Learner.clairvoyant import *
from plotting.plot_reward_regret import *
from resources.NSEnvironment import NSEnvironment

mean_iteration = 2
def iterate(learner, env, iteration, daily_simulation, clairvoyant_price_index, name_alg, n_step=3):

    global price_pulled, reward_observed, product_visited, items_bought, items_rewards, clairvoyant_margin_values

    cumulative_reward = np.zeros((mean_iteration, iteration))
    cumulative_regret = np.zeros((mean_iteration, iteration))
    final_reward = np.zeros((mean_iteration, iteration))

    for z in range(mean_iteration):
        product_visited_list = []
        items_bought_list = []
        print("plot ite: ", z)
        # re-initialized lambda, secondary prod, user classes, no of arms
        learner.reset()

        for i in range(iteration):
            print("iteration: ", i)
            learner.debug()
            price_pulled = learner.act()
            reward_observed, product_visited, items_bought, items_rewards = env.round(price_pulled)
            product_visited_list += product_visited[0]
            items_bought_list += items_bought[0]
            learner.update(price_pulled, reward_observed, product_visited, items_bought, items_rewards)
            learner.update_pulled_and_success(price_pulled,
                                              product_visited_list, items_bought_list, items_rewards)

        match n_step:
            case 3 | 4 | 5:
                clairvoyant_margin_value = find_clairvoyant_reward(learner, env, clairvoyant_price_index, daily_simulation)
                #clairvoyant_margin_values = np.repeat(clairvoyant_margin_value, iteration).astype(np.float64)
                if debug:
                    print("\nReward clairvoyant: ", clairvoyant_margin_values)
                clairvoyant_margin_iterated = np.full(iteration, clairvoyant_margin_value)
                clairvoyant_margin_values = clairvoyant_margin_iterated
                # TODO to be adapted with changes
                changes = []
            case 6:
                assert isinstance(env, NSEnvironment)
                env_test = copy.deepcopy(env)
                clairvoyant_margin_iterated = np.zeros(shape=iteration)

                t = 0
                changes_instants = list(env_test.changes_instant)
                changes_instants.append(iteration)

                for instant,users in env_test.changes_collector:
                    # TODO: clairvoyant_price_index ?
                    # 1 function is overwrite by manual setting
                    # 2 should the solution change for different conv rates ? yes it should ..
                    # clairvoyant_price_index = ...
                    t_change = changes_instants.pop(0)
                    env_test.users = users
                    #clairvoyant_margin_value = find_clairvoyant_reward(learner, env_test, clairvoyant_price_index, daily_simulation)
                    clairvoyant_indexes, clairvoyant_margin_value = find_clairvoyant_reward_by_simulation(env_test)
                    clairvoyant_margin_value = (clairvoyant_margin_value + learner.top_reward) / 2
                    clairvoyant_margin_iterated[t:t_change] = np.repeat(clairvoyant_margin_value, t_change-t)
                    t = t_change
                clairvoyant_margin_values = clairvoyant_margin_iterated
                changes = env.changes_instant

        cumulative_reward[z, :] = np.cumsum(learner.list_margins)
        cumulative_regret[z, :] = np.cumsum(clairvoyant_margin_iterated) - np.cumsum(learner.list_margins)
        final_reward[z, :] = learner.list_margins

    return cumulative_regret, cumulative_reward, final_reward, clairvoyant_margin_values

def compute_ratio_regret(learner, clairvoyant_values, final_rewards, cumulative_regret):
    ### COMPUTE RATIO REGRET
    clairvoyant_values = np.mean(clairvoyant_values, axis=0)
    final_rewards = np.mean(final_rewards, axis=0)
    delta_reward = np.subtract(clairvoyant_values, final_rewards)
    upper_bound_regret_list = []
    c1 = 4 * np.log(learner.t)
    for val in delta_reward:
        if val > 0:
            x = (c1 / val) + (8 * val)
            upper_bound_regret_list.append(x)
    upper_bound_regret = np.sum(np.array(upper_bound_regret_list))
    print("cumulative regret : ", cumulative_regret[-1])
    print("upper bound regret : ", upper_bound_regret)
    cumulative_regret = np.mean(cumulative_regret, axis=0)
    ratio = cumulative_regret[-1] / upper_bound_regret

    return ratio
