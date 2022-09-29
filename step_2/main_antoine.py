import numpy.random as npr
from simulator import *
from users import *
import matplotlib.pyplot as plt
from copy import deepcopy


def website_simulation(users,simulator):
# This method simulates users visiting the ecommerce website
# argument is an User class instance
# returns total rewards for all five products

    total_rewards=np.zeros(5,np.float16)

    for j in range(len(users.n_users)-1):
        
        for n in range(round(users.n_users[j+1])):
            simulator.visited_primaries=[]
            rewards=simulator.simulation(j,users)
            total_rewards += rewards
    return total_rewards
    

def algorithm_1(users):
    prices_ind=[0,0,0,0,0]
    sim = Simulator(prices_ind,margins,lamb)
    previous=0
    max_reward=np.sum(website_simulation(users,sim))
    max_prices_bool=np.zeros(5)
    ite=0
    price_history=[]
    reward_history=[]

    while max_reward > previous and np.sum(max_prices_bool)<5:
        if ite>0:
            prices_ind[np.argmax(sum_rewards)]+=1
        reward_history.append(max_reward)      
        price_history.append(deepcopy(prices_ind))

        previous = max_reward
        sum_rewards=np.zeros(5)

        #try each possible price increase
        for i in range(5):
            if prices_ind[i]<3:
            #test if maximum is reached
                prices_ind[i]+=1
                sum_rewards[i]=np.sum(website_simulation(users,sim))
                prices_ind[i]-=1
            else :
                max_prices_bool[i]=True
        max_reward=np.max(sum_rewards)
        if max_reward > previous:
            ite+=1

    return price_history,reward_history,ite

if __name__=="__main__":

    # PARAMETERS TO CHOOSE ARBITRARILY
    alpha_ratios=npr.dirichlet([50,10,10,10,10,50])
    total_users=npr.normal(500,10)
    graph_weights=npr.random((5,5))
    v=np.array([3,3,2,1]).reshape((4,1))
    n_items_bought=np.int64(npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v)))
    v=np.array([0.8,0.6,0.4,0.2]).reshape((4,1))
    conv_rates=npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v))
    users_A= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
    margins=npr.random((4,5))*20
    lamb=0.2

    results=algorithm_1(users_A)
    print(results[0])
    print(results[1])
    print(results[2])

    plt.figure(0)
    plt.title("greedy algorithm")
    plt.plot([str(el) for el in results[0]], results[1], label = "rewards decision")
    plt.xlabel("price")
    plt.xticks(rotation='vertical')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel("margin")

    plt.legend()
    plt.show()