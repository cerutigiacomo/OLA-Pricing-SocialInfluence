import json
import matplotlib.pyplot as plt

f = open('../resources/environment.json')
data = json.load(f)

different_value_of_prices = data["product"]["different_value_of_prices"]
numbers_of_products = data["product"]["numbers_of_products"]
users_classes = len(data["users"]["classes"])
colors = ["r", "g", "b"]


def plot_simulator(margins, prices, secondary):
    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(list(range(numbers_of_products)), margins.transpose(), 'bo', mfc='none')
    for i in range(numbers_of_products):
        axs[0].plot(i, margins.transpose()[i][prices[i]], 'rx')
    axs[0].set_title("Price of the product selected")
    axs[0].grid()
    axs[1].plot(list(range(numbers_of_products)), secondary, 'bo', mfc='none')
    axs[1].set_title("Secondary association")
    axs[1].set_yticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
    axs[1].sharex(axs[0])
    axs[1].grid()
    plt.xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
    fig.tight_layout()
    plt.show()


def plot_users(total_users, alpha_ratios, graph, n_items_bought, prices, max_item_bought, conv_rates):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
    axs[0, 0].bar(['Class A', 'Class B', 'Class C'], total_users)
    axs[0, 0].set_xticks(list(range(users_classes)), list(range(1, 1 + users_classes)))
    axs[0, 0].set_title("Total users")
    axs[0, 0].grid()
    axs[0, 0].set_ylim([0, data["simulator"]["max_users"]])

    for i in range(users_classes):
        axs[0, 1].plot(list(range(1 + numbers_of_products)), alpha_ratios[i], colors[i] + 'o', mfc='none',
                       label='class' + str(i))
    axs[0, 1].set_xticks(list(range(1 + numbers_of_products)), list(range(1 + numbers_of_products)))
    axs[0, 1].set_title("Alpha Ratios")
    axs[0, 1].grid()
    axs[0, 1].set_ylim([0, 1])

    for i in range(users_classes):
        axs[1, i].plot(graph[i], colors[i] + 'o', mfc='none', label='class' + str(i))
        axs[1, i].set_xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
        axs[1, i].set_title("Graph w. " + str(i))
        axs[1, i].grid()
        axs[1, i].set_ylim([0, 1])

    for i in range(users_classes):
        items = [n_items_bought[i].transpose()[j][prices[j]] for j in range(numbers_of_products)]
        axs[2, i].plot(items, colors[i] + 'o-', mfc='none',
                       label='class' + str(i))

        axs[2, i].set_xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
        axs[2, i].set_yticks(list(range(1 + max_item_bought * 2)), list(range(1 + max_item_bought * 2)))
        axs[2, i].set_title("N items bought " + str(i))
        axs[2, i].grid()
        axs[2, i].set_ylim([0, max_item_bought])

    for i in range(users_classes):
        axs[i, 3].plot(conv_rates[i], colors[i] + 'o-', mfc='none', label='class' + str(i))
        axs[i, 3].set_xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
        axs[i, 3].set_title("Conv. Rates " + str(i))
        axs[i, 3].grid()
        axs[i, 3].set_ylim([0, 1])

    # a = n_items_bought[0].transpose()
    fig.tight_layout()
    plt.show()


def plot_reward(reward):
    plt.rcParams["figure.figsize"] = (15,10)
    plt.plot(list(range(numbers_of_products)), reward, 'o-', mfc='none')
    plt.xlabel('Product')
    plt.ylabel('Reward')
    plt.xticks(list(range(numbers_of_products)), list(range(numbers_of_products)))
    plt.grid()
    plt.suptitle('Rewards')

    plt.show()