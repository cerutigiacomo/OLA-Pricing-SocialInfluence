import matplotlib.pyplot as plt
import numpy as np
import json

f = open('../resources/environment.json')
data = json.load(f)
days = data["simulator"]["days"]

def mean_std(data, day=days):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) / np.sqrt(day)
    # TODO iterate the learner more times and get the mean of the results!
    # return mean, std
    return data, 0

def plot_regret_reward(cumulative_regret,
                       cumulative_reward,
                       final_reward,
                       best_revenue_array, best_revenue,
                       label_alg,
                       day=days):

    mean_cumulative_regret, stdev_regret = mean_std(cumulative_regret, day)
    mean_cumulative_reward, stdev_cumulative_reward = mean_std(cumulative_reward, day)
    mean_final_reward, stdev_reward = mean_std(final_reward, day)

    fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(20,20))
    ax[0].plot(mean_cumulative_regret, color='blue', label=label_alg)
    #ax[0].fill_between(range(days), mean_cumulative_regret - stdev_regret,mean_cumulative_regret + stdev_regret, alpha=0.4)
    ax[0].set_title('Cumulative Regret')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(mean_final_reward, color='blue', label=label_alg)
    #ax[1].fill_between(range(days), mean_final_reward - stdev_reward, mean_final_reward + stdev_reward, alpha=0.4)
    ax[1].axhline(y=best_revenue, color='red', linestyle='--', label='Clairvoyant')
    ax[1].set_title('Reward')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(mean_cumulative_reward, color='blue', label=label_alg)
    #ax[2].fill_between(range(days), mean_cumulative_reward - stdev_cumulative_reward, mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
    ax[2].plot(np.cumsum(best_revenue_array), color='red', linestyle='--', label='Clairvoyant')
    ax[2].set_title('Cumulative reward')
    ax[2].legend()
    ax[2].grid()

    plt.show()