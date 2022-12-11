import json
import matplotlib.pyplot as plt

f = open('../resources/environment.json')
data = json.load(f)

different_value_of_prices = data["product"]["different_value_of_prices"]
numbers_of_products = data["product"]["numbers_of_products"]
users_classes = len(data["users"]["classes"])
colors = ["r", "g", "b", "y"]
classes = data["users"]["classes"]
max_item_bought = data["simulator"]["max_item_bought"]


def plot_simulator(sim):
    margins = sim.margins
    prices = sim.prices
    secondary = sim.secondary_product

    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(list(range(numbers_of_products)), margins, 'o', label="margins")
    axs[0].plot(list(range(numbers_of_products)), prices, 'x', label='prices')
    axs[0].set_title("Price and margin of the product selected")
    axs[0].grid()
    axs[1].plot(list(range(numbers_of_products)), secondary, 'bo', mfc='none')
    axs[1].set_title("Secondary association")
    axs[1].set_yticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
    axs[1].sharex(axs[0])
    axs[1].grid()
    plt.xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
    fig.tight_layout()
    plt.show()


def plot_users(users, classes_idx):
    total_users = [users[i].total_users for i in range(len(classes_idx))]
    alpha_ratios = [users[i].alpha for i in range(len(classes_idx))]
    graph = [users[i].graph_weights for i in range(len(classes_idx))]
    n_items_bought = [users[i].n_items_bought for i in range(len(classes_idx))]
    conv_rates = [users[i].conv_rates for i in range(len(classes_idx))]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), gridspec_kw={'width_ratios': [2, 2, 2]})
    axs[0, 0].bar(['Class ' + str(i) for i in range(len(classes_idx))], total_users)
    axs[0, 0].set_xticks(list(range(len(classes_idx))), list(range(1, 1 + len(classes_idx))))
    axs[0, 0].set_title("Total users")
    axs[0, 0].grid()
    axs[0, 0].set_ylim([0, data["simulator"]["max_users"]])

    for i in range(len(classes_idx)):
        axs[0, 1].plot(list(range(1 + numbers_of_products)), alpha_ratios[i], colors[i] + 'o-', mfc='none',
                       label='class' + classes[classes_idx[i]]["name"])
    axs[0, 1].set_xticks(list(range(1 + numbers_of_products)), list(range(1 + numbers_of_products)))
    axs[0, 1].set_title("Alpha Ratios")
    axs[0, 1].grid()
    axs[0, 1].set_ylim([0, 1])

    for i in range(len(classes_idx)):
        axs[0, 2].plot(n_items_bought[i], colors[i] + 'o-', mfc='none',
                       label='class' + classes[classes_idx[i]]["name"])

        axs[0, 2].set_xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
        axs[0, 2].set_yticks(list(range(1 + max_item_bought * 2)), list(range(1 + max_item_bought * 2)))
        axs[0, 2].set_title("N items bought ")
        axs[0, 2].grid()
        axs[0, 2].legend()
        axs[0, 2].set_ylim([0, max_item_bought])

    for i in range(len(classes_idx)):
        for j in range(different_value_of_prices):
            axs[1, i].plot(conv_rates[i].transpose()[j], colors[j] + 'o-', mfc='none', label=str(i))
        axs[1, i].set_xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
        axs[1, i].set_title("Conv. Rates " + classes[classes_idx[i]]["name"])
        axs[1, i].grid()
        axs[1, i].set_ylim([0, 1])

    # a = n_items_bought[0].transpose()
    fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    for i in range(len(classes_idx)):
        for j in range(numbers_of_products):
            axs[j].plot(graph[i][j], colors[i] + 'o-', mfc='none', label=classes[classes_idx[i]]["name"])
            axs[j].set_xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
            axs[j].set_title("Graph weights pr:" + str(j))
            axs[j].grid()
            axs[j].set_ylim([0, 1])
    fig.tight_layout()
    plt.show()


def plot_reward(reward):
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.plot(list(range(numbers_of_products)), reward, 'o-', mfc='none')
    plt.xlabel('Product')
    plt.ylabel('Reward')
    plt.xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
    plt.grid()
    plt.suptitle('Rewards')

    plt.show()


def plot_result_simulation(reward, visited):
    plt.rcParams["figure.figsize"] = (50, 10)
    plt.plot(reward, 'o-', mfc='none')
    plt.xticks(list(range(len(reward))), visited,  rotation='vertical')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.grid()

    plt.show()

