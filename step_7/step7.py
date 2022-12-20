# TODO use the simulator to simulate next reward and not the website simulator, to reduce computation
from clairvoyant import *
from plotting.plot_reward_regret import plot_regret_reward_split_classes
from step_7.Environment import Environment
from step_7.Context.ContextGenerator import ContextGenerator
from step_7.Context.ContextNode import ContextNode
from step_7.Context.ContextualLearner import ContextualLearner
from step_7.TS_7_Learner import TSLearner
from step_7.UCB_7_Learner import *
from step_7.utils import get_right_user_class

f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]
features = data["users"]["features"]
debug = True
PLOT_ITERATION = 1
DAYS = 100
SIMULATION_ITERATIONS = 10
ENVIRONMENT_ITERATIONS = 4

# Set as False for running the simulation on TS!
UCB_LEARNER = True


final_reward= np.zeros((PLOT_ITERATION, DAYS))
final_cumulative_regret = np.zeros((PLOT_ITERATION, DAYS))
final_cumulative_reward = np.zeros((PLOT_ITERATION, DAYS))

prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA

for k in range (PLOT_ITERATION):
    users_classes_to_import = [0, 1, 2]
    env = Environment(different_value_of_prices, prices, margins,
                      [0, 0, 0, 0, 0], users_classes_to_import, ENVIRONMENT_ITERATIONS)
    learner = Learner(different_value_of_prices)

    best_prices_per_class = [[3, 3, 3, 1, 3], [3, 3, 2, 3, 2], [3, 3, 3, 3, 3]]
    best_not_aggregated_reward = find_not_aggregated_reward(best_prices_per_class, env)
    print("best_not_aggregated_reward: ", best_not_aggregated_reward)
    context_learner = ContextualLearner(features=features, n_arms=env.n_arms, n_products=numbers_of_products)
    if UCB_LEARNER:
        root_learner = UCBLearner(users_classes_to_import, different_value_of_prices)
    else:
        root_learner = TSLearner(users_classes_to_import, different_value_of_prices, DAYS)

    root_node = ContextNode(features=features, base_learner=root_learner)
    context_learner.update_context_tree(root_node)
    context_generator = ContextGenerator(features=features,
                                         contextual_learner=context_learner,
                                         users_classes_to_import=users_classes_to_import,
                                         days_to_simulate=DAYS,
                                         confidence=0.1,
                                         iteration=SIMULATION_ITERATIONS)

    opt_rew = []
    actual_rew = [] # TODO size and populate the array here!

    for i in range(DAYS):
        print("Iteration " + str(k) + ": day " + str(i))

        if i % 14 == 0 and i != 0:
            context_generator.context_generation()
        rew = np.zeros(numbers_of_products)
        pulled_arms = np.zeros(numbers_of_products)
        for j in range(SIMULATION_ITERATIONS):
            current_features, name_features = get_right_user_class(users_classes_to_import)
            learner = context_learner.get_learner_by_context(current_features=name_features)

            pulled_arms = learner.act()
            rew, visited_products, num_bought_products, num_primary = env.round(pulled_arms, current_features)
            # print("current_features: ", current_features, "rew: ", np.sum(rew))
            num_primary = 0 # TODO remove also from the context generator!
            # num_primary: number
            learner.updateHistory(rew, pulled_arms, visited_products, num_bought_products, num_primary)
            context_generator.collect_daily_data(pulled_arms=pulled_arms.copy(),
                                                 coll_rewards=rew,
                                                 visited_products=visited_products,
                                                 num_bought_products=num_bought_products[0],
                                                 num_primaries=num_primary,
                                                 features=name_features)
        learner.update(pulled_arms.copy(), rew, visited_products, num_bought_products)
        context_generator.update_average_rewards(current_features=name_features)

        if debug:
            print("context_last_iterations_mean_rew rewards: ", context_generator.average_rewards[-1],
                  " rew: ", np.sum(rew),
                  " Pulled arms: ", pulled_arms,
                  "name feature: ", context_generator.collected_features[-1])

        actual_rew.append(np.sum(context_generator.average_rewards[-1]))
        opt_rew.append(best_not_aggregated_reward)

    final_cumulative_reward[k, :] = np.cumsum(actual_rew)
    final_cumulative_regret[k, :] = np.cumsum(opt_rew) - np.cumsum(actual_rew)
    final_reward[k:] = actual_rew

name = "step_7_UCB" if UCB_LEARNER else "step_7_TS"

# Plot Regret and Reward
plot_regret_reward_split_classes(final_cumulative_regret,
                   final_cumulative_reward,
                   final_reward,
                   best_not_aggregated_reward,
                   label_alg=name,
                   day=DAYS)
