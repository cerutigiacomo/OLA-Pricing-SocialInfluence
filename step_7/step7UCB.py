# TODO use the simulator to simulate next reward and not the website simulator, to reduce computation
from clairvoyant import *
from plotting.plot_reward_regret import plot_regret_reward_split_classes
from step_7.Environment import Environment
from step_7.ContextGenerator import ContextGenerator
from step_7.ContextNode import ContextNode
from step_7.ContextualLearner import ContextualLearner
#from step_7.TS_7_Learner import *
from step_7.UCB_7_Learner import *
from step_7.choice_user_class import get_right_user_class
import math

f = open('../resources/environment.json')
data = json.load(f)
max_item_bought = data["simulator"]["max_item_bought"]
features = data["users"]["features"]
debug = False
class_choosed = [0]
PLOT_ITERATION = 5
DAYS = 100
SIMULATION_ITERATIONS = 5



final_reward= np.zeros((PLOT_ITERATION, DAYS))
final_cumulative_regret = np.zeros((PLOT_ITERATION, DAYS))
final_cumulative_reward = np.zeros((PLOT_ITERATION, DAYS))

prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA

for k in range (PLOT_ITERATION):
    users_classes_to_import = [0, 1, 2]
    env = Environment(different_value_of_prices, prices, margins,
                      [0, 0, 0, 0, 0], users_classes_to_import)
    learner = Learner(different_value_of_prices)

    best_index = [3, 3, 3, 3, 3]
    best_reward = find_clairvoyant_reward(learner,
                                          env, best_index, SIMULATION_ITERATIONS)
    best_reward_array = [best_reward for _ in range(DAYS)]

    best_prices_per_class = [[3, 3, 3, 1, 3], [3, 3, 1, 3, 2], [3, 3, 1, 1, 3]]
    best_not_aggregated_reward = find_not_aggregated_reward(best_prices_per_class, env)

    context_learner = ContextualLearner(features=features, n_arms=env.n_arms, n_products=numbers_of_products)
    root_learner = UCBLearner(users_classes_to_import, 4)

    root_node = ContextNode(features=features, base_learner=root_learner)
    context_learner.update_context_tree(root_node)

    # confidence used for lower bounds is hardcoded to 0.1!
    context_generator = ContextGenerator(features=features,
                                         contextual_learner=context_learner,
                                         users_classes_to_import=users_classes_to_import,
                                         confidence=0.1,
                                         iteration=SIMULATION_ITERATIONS)

    opt_rew = []
    actual_rew = []

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
            num_primary = 0
            # num_primary: number
            learner.updateHistory(rew, pulled_arms, visited_products, num_bought_products, num_primary)
            num_bought_products_summed = np.sum(np.array(num_bought_products[0]), axis = 0)
            context_generator.collect_daily_data(pulled_arms=pulled_arms,
                                                 coll_rewards=rew,
                                                 visited_products=visited_products,
                                                 num_bought_products=num_bought_products_summed,
                                                 num_primaries=num_primary,
                                                 features=name_features)

        learner.update(pulled_arms, rew, visited_products, num_bought_products)
        context_generator.update_average_rewards(current_features=name_features)

        if context_generator.average_rewards[-1] > 700:
            print("context_generator rewards OVER 400: ", context_generator.average_rewards[-1],
                  " Pulled arms: ", pulled_arms,
                  " rew: ", np.sum(rew),
                  "name feature: ", get_name_feature(name_features))
        else:
            print("context_generator rewards: ", context_generator.average_rewards[-1],
                  " Pulled arms: ", pulled_arms,
                  " rew: ", np.sum(rew),
                  "name feature: ", get_name_feature(name_features))

        actual_rew.append(context_generator.average_rewards[-1])
        opt_rew.append(best_not_aggregated_reward)

    final_cumulative_regret[k, :] = np.cumsum(opt_rew) - np.cumsum(actual_rew)
    final_cumulative_reward[k, :] = np.cumsum(actual_rew)
    final_reward[k:] = actual_rew


# Plot Regret and Reward
plot_regret_reward_split_classes(final_cumulative_regret,
                   final_cumulative_reward,
                   final_reward,
                   best_reward,
                   label_alg="step_7_UCB",
                   day=DAYS)