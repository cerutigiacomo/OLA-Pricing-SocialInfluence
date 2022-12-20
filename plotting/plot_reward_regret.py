import matplotlib.pyplot as plt
import numpy as np
import json

f = open('../resources/environment.json')
data = json.load(f)
days = data["simulator"]["days"]

def mean_std(data_, day=days):
    mean = np.mean(data_, axis=0)
    std = np.std(data_, axis=0) / np.sqrt(day)

    return mean, std

def plot_regret_reward(cumulative_regret,
                       cumulative_reward,
                       final_reward,
                       best_revenue,
                       changes,
                       label_alg,
                       day=days):

    best_revenue_array = np.repeat(best_revenue, day).astype(np.float64)
    #best_revenue_array = np.max(best_revenue,axis=0)

    mean_cumulative_regret, stdev_regret = mean_std(cumulative_regret, day)
    mean_cumulative_reward, stdev_cumulative_reward = mean_std(cumulative_reward, day)
    mean_reward, stdev_reward = mean_std(final_reward, day)

    fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(12,8))
    ax[0].plot(mean_cumulative_regret, color='blue', label=label_alg)
    ax[0].fill_between(range(day), (mean_cumulative_regret - stdev_regret),(mean_cumulative_regret + stdev_regret), alpha=0.4)
    ax[0].set_title('Cumulative Regret')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(mean_reward, color='blue', label=label_alg)
    ax[1].fill_between(range(day), mean_reward - stdev_reward, mean_reward + stdev_reward, alpha=0.4)
    ax[1].plot(best_revenue_array, color='red', linestyle='--', label='Clairvoyant')
    for change in changes:
        ax[1].axvline(change, color="darkviolet", lw=3)
    ax[1].set_title('Reward')
    ax[1].legend()
    ax[1].grid()



    ax[2].plot(mean_cumulative_reward, color='blue', label=label_alg)
    ax[2].fill_between(range(day), mean_cumulative_reward - stdev_cumulative_reward, mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
    ax[2].plot(np.cumsum(best_revenue_array), color='red', linestyle='--', label='Clairvoyant')
    ax[2].set_title('Cumulative reward')
    ax[2].legend()
    ax[2].grid()

    plt.show()

def plot_regret_reward_split_classes(cumulative_regret,
                       cumulative_reward,
                       final_reward,
                       best_revenue,
                       label_alg,
                       day=days):
    best_revenue_array = np.repeat(best_revenue, day).astype(np.float64)
    mean_cumulative_regret, stdev_regret = mean_std(cumulative_regret, day)
    mean_cumulative_reward, stdev_cumulative_reward = mean_std(cumulative_reward, day)
    mean_reward, stdev_reward = mean_std(final_reward, day)

    fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(12,8))
    ax[0].plot(mean_cumulative_regret, color='blue', label=label_alg)
    ax[0].fill_between(range(day), (mean_cumulative_regret - stdev_regret),(mean_cumulative_regret + stdev_regret), alpha=0.4)
    ax[0].set_title('Cumulative Regret')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(mean_reward, color='blue', label=label_alg)
    ax[1].fill_between(range(day), mean_reward - stdev_reward, mean_reward + stdev_reward, alpha=0.4)
    ax[1].axhline(y=best_revenue, color='red', linestyle='--', label='Clairvoyant')
    ax[1].axvline(x=14, color='red', label="Split attempt")
    ax[1].axvline(x=28, color='red')
    ax[1].axvline(x=42, color='red')
    ax[1].axvline(x=56, color='red')
    ax[1].axvline(x=70, color='red')
    ax[1].axvline(x=84, color='red')
    ax[1].axvline(x=98, color='red')
    ax[1].set_xticks(np.arange(0, 101, 5))
    ax[1].set_title('Reward')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(mean_cumulative_reward, color='blue', label=label_alg)
    ax[2].fill_between(range(day), mean_cumulative_reward - stdev_cumulative_reward, mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
    ax[2].plot(np.cumsum(best_revenue_array), color='red', linestyle='--', label='Clairvoyant')
    ax[2].set_title('Cumulative reward')
    ax[2].legend()
    ax[2].grid()

    plt.show()

def plot_regret_reward_UCB6(cumulative_regret,
                            cumulative_regretSW,
                            cumulative_regret_first,
                            cumulative_regret_second,
                            cumulative_reward,
                            cumulative_rewardSW,
                            final_reward,
                            final_rewardSW,
                            best_revenue,
                            best_revenue_new,
                            label_alg,
                            day=days):
    best_revenue_array = np.repeat(best_revenue,day/2).astype(np.float64)
    best_revenueAC_array = np.concatenate([best_revenue_array, np.repeat(best_revenue_new, day/2).astype(np.float64)])

    mean_cumulative_regret, stdev_regret = mean_std(cumulative_regret, day)
    mean_cumulative_regretSW, stdev_regret = mean_std(cumulative_regretSW, day)
    mean_cumulative_regret_first, stdev_regret_first = mean_std(cumulative_regret_first, day/2)
    mean_cumulative_regret_second, stdev_regret_second = mean_std(cumulative_regret_second, day/2)
    mean_cumulative_reward, stdev_cumulative_reward = mean_std(cumulative_reward, day)
    mean_cumulative_rewardSW, stdev_cumulative_rewardSW = mean_std(cumulative_rewardSW, day)
    mean_reward, stdev_reward = mean_std(final_reward, day)
    mean_rewardSW, stdev_reward = mean_std(final_rewardSW, day)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

    ax[0].plot(mean_cumulative_regret, color='blue', label="UCB")
    ax[0].plot(mean_cumulative_regretSW, color='red', label="SW UCB")
    #ax[0].plot(mean_cumulative_regret_first, color='red', label=label_alg)
    #ax[0].plot(mean_cumulative_regret_second, color='green', label=label_alg)
    #ax[0].plot(mean_cumulative_regret_first, color='red', label=label_alg)
    #ax[0].plot(mean_cumulative_regret_second, color='green', label=label_alg)
    ax[0].fill_between(range(day), (mean_cumulative_regret - stdev_regret), (mean_cumulative_regret + stdev_regret), alpha=0.4)
    #ax[0].axhline(y=mean_cumulative_regret_first, color='red', xmax=0.5, linestyle='--', label='Clairvoyant')
    #ax[0].axhline(y=mean_cumulative_regret_second, color='green', xmin=0.5, linestyle='--', label='After AC' )
    ax[0].set_title('Cumulative Regret')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(mean_reward, color='blue', label="UCB")
    ax[1].plot(mean_rewardSW, color='red', label="SW UCB")
    # point for the abrupt change
    # ax[1].plot(0, 25, marker="o", color="red")
    ax[1].fill_between(range(day), mean_reward - stdev_reward, mean_reward + stdev_reward, alpha=0.4)
    ax[1].axhline(y=best_revenue, color='red', xmax=0.5, linestyle='--', label='Clairvoyant')
    ax[1].axhline(y=best_revenue_new, color='green', xmin=0.5, linestyle='--', label='ClairvoyantAC')
    ax[1].set_title('Reward')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(mean_cumulative_reward, color='blue', label="UCB")
    ax[2].plot(mean_cumulative_rewardSW, color='red', label="SW UCB")
    ax[2].fill_between(range(day), mean_cumulative_reward - stdev_cumulative_reward,
                       mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
    ax[2].plot(np.cumsum(best_revenue_array), color='red', linestyle='--', label='Clairvoyant')
    ax[2].plot(np.cumsum(best_revenueAC_array), color='green', linestyle='--', label='ClairvoyantAC')
    ax[2].set_title('Cumulative reward')
    ax[2].legend()
    ax[2].grid()

    plt.show()

def plot_regret_reward_step6(cumulative_regret,
                       cumulative_reward,
                       final_reward,
                       best_revenue,
                       changes,
                       label_alg,
                       color_alg,
                       ax,
                       day=days):
    best_revenue_array = np.max(best_revenue, axis=0)

    mean_cumulative_regret, stdev_regret = mean_std(cumulative_regret, day)
    mean_cumulative_reward, stdev_cumulative_reward = mean_std(cumulative_reward, day)
    mean_reward, stdev_reward = mean_std(final_reward, day)

    ax[0].plot(mean_cumulative_regret, color=color_alg, label=label_alg)
    ax[0].fill_between(range(day), (mean_cumulative_regret - stdev_regret), (mean_cumulative_regret + stdev_regret),
                       alpha=0.4)
    ax[0].set_title('Cumulative Regret')

    ax[1].plot(mean_reward, color=color_alg, label=label_alg)
    ax[1].fill_between(range(day), mean_reward - stdev_reward, mean_reward + stdev_reward, alpha=0.4)
    # ax[1].plot(best_revenue, color='red', linestyle='--', label='Clairvoyant')
    ax[1].step(np.arange(0, best_revenue_array.shape[0]), best_revenue_array, color='red', linestyle="--",
               label='Clairvoyant')
    for change in changes:
        ax[1].axvline(change - 1, color="darkviolet", lw=3)
    ax[1].set_title('Reward')

    ax[2].plot(mean_cumulative_reward, color=color_alg, label=label_alg)
    ax[2].fill_between(range(day), mean_cumulative_reward - stdev_cumulative_reward,
                       mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
    ax[2].plot(np.cumsum(best_revenue_array), color='red', linestyle='--', label='Clairvoyant')
    ax[2].set_title('Cumulative reward')
