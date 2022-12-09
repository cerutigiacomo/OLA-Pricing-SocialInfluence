import matplotlib.pyplot as plt
def plot_greedy(prices, margins):
    plt.figure(0)
    plt.plot(prices, margins, label="rewards decision")
    plt.xlabel("price")
    plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel("margin")
    plt.grid()

    plt.legend()
    plt.show()

def plot_reward_comparison(name,data):
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(list(name), data, color='maroon', width=0.4)

    plt.xlabel("Algorithms")
    plt.ylabel("Reward")
    plt.title("Reward comparison")
    plt.grid()
    plt.show()
