from msilib.schema import Error
from numpy import argmax, int16
import numpy.random as npr
from simulator import *
from users import *
import matplotlib.pyplot as plt
from copy import deepcopy

prep=8
iterations=30

def website_simulation(users,simulator):
# This method simulates users visiting the ecommerce website
# argument is an User class instance
# returns total rewards for all five products

    total_rewards=np.zeros(5,np.float64)

    for j in range(len(users.n_users)-1):
        for n in range(round(users.n_users[j+1])):
            simulator.visited_primaries=[]
            # users.graph_weights=npr.random((5,5))
            # alpha_ratios=npr.dirichlet([50,10,10,10,10,50])
            # total_users=npr.normal(500,10)
            # users.n_users=alpha_ratios*total_users
            # users.n_items_bought=npr.randint(1,20,(4,5))
            # users.conv_rates=np.maximum(0.1,npr.random((4,5)))

            rewards=simulator.simulation(j,users)
            total_rewards += rewards
    return total_rewards
    
def update_std(t,nc):
    # second term of the upper confidence
    std=np.sqrt(2*np.log(t)/nc)
    return std

def algorithm_1(users,n_ite):
    if n_ite<=prep:
        return 0
    visited_configr={}
    ite=0
    price_history=[]
    reward_history=[]
    total_reward_history=np.zeros(n_ite)
    accumulated_rewards=np.zeros((4,5))
    nc=np.zeros((4,5))

    prices_ind=[0,0,0,0,0]
    sim = Simulator(prices_ind,margins,lamb)

    for i in range(8):
    #play each arm twice
        j=i%4
        sim.prices=[j,j,j,j,j]
        # prices_ind=[j,j,j,j,j]
        rewards=website_simulation(users,sim)
        nc[j]+=np.ones(5)
        accumulated_rewards[j]+=rewards
        ite+=1
        
        total_reward_history[ite-1]=np.sum(rewards)
        reward_history.append(deepcopy(rewards))      
        price_history.append(deepcopy(sim.prices))

    mean_reward=accumulated_rewards/nc
    max_of_mean=np.max(mean_reward,axis=0)
    min_of_mean=np.min(mean_reward,axis=0)
    mean_normalized=(mean_reward-min_of_mean)/(max_of_mean-min_of_mean)
    std=update_std(ite+1,nc)
    upper=std+mean_normalized
    
    for i in range(n_ite-prep):
    # perform the algorithm for the ramining iterations

        #select the ind with biggest upper confidence
        prices_ind_array=np.argmax(upper,axis=0)
        prices_ind=[ind for ind in prices_ind_array]
        sim.prices=prices_ind
        rewards=website_simulation(users,sim)

        #register configuration and increment visitation to compute clairvoyant later
        if str(sim.prices) in visited_configr:
            visited_configr[str(sim.prices)]+=1
        else: visited_configr[str(sim.prices)]=1

        #compute and upadte upper confidence
        for k in range(len(rewards)):
            nc[prices_ind[k],k]+=1
            accumulated_rewards[prices_ind[k],k]+=rewards[k]
        mean_reward=accumulated_rewards/nc
        max_of_mean=np.max(np.array(reward_history[0:prep]),axis=0)
        min_of_mean=np.min(np.array(reward_history[0:prep]),axis=0)
        mean_normalized=(mean_reward-min_of_mean)/(max_of_mean-min_of_mean)
        ite+=1
        std=update_std(ite+1,nc)
        upper=std+mean_normalized
        total_reward_history[ite-1]=np.sum(rewards)
        reward_history.append(deepcopy(rewards))      
        price_history.append(deepcopy(prices_ind))

    return price_history,reward_history,ite,upper,std,mean_reward,visited_configr,total_reward_history

if __name__=="__main__":

    ######## PARAMETERS TO CHOOSE ARBITRARILY #####
    alpha_ratios=npr.dirichlet([50,10,10,10,10,50])
    total_users=npr.normal(500,10)
    graph_weights=npr.random((5,5))
    # v=npr.random(4).reshape((4,1))
    # n_items_bought=np.maximum(1,np.int64(npr.normal(20,3,(4,5))*np.hstack((v,v,v,v,v))))
    n_items_bought=npr.randint(1,20,(4,5))
    # v=npr.random(4).reshape((4,1))
    # conv_rates=np.minimum(1,np.maximum(0.1,npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v))))
    conv_rates=np.maximum(0.1,npr.random((4,5)))
    users_A= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
    margins=npr.random((4,5))*20
    lamb=0.2
    ##############################################

    results=algorithm_1(users_A,iterations)

    # determine the most recurrent price configuration and select as optimal one
    max=-np.inf
    for item in results[6].items():
        if item[1]>max:
            max=item[1]
            max_config=item[0].strip('[]').split(',')
    for m in range(len(max_config)):
        max_config[m]=np.int32(max_config[m])
    print("max config \n",max_config)
    print("mean \n",results[5])

    # plot reward over time
    plt.figure(0)
    plt.title("Rewards over time per product")
    plt.plot(np.arange(len(results[1])), results[1])
    plt.xlabel("iterations")
    plt.ylabel("rewards")
    #plot the regret
    plt.figure()
    plt.plot(np.cumsum(results[7])/np.arange(iterations)+1,label="expected reward")
    plt.axhline(y = np.sum(results[5][max_config,np.arange(results[5].shape[1])]), color = 'r', linestyle = '-',label = 'clairvoyant optimal sum of rewards')
    plt.title("Regret of the Bandit algorithm N=" + str(iterations))
    plt.legend()
    plt.show()