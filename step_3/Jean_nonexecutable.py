import numpy.random as npr
from simulator import *
from users import *
from website_simulation import website_simulation
import matplotlib.pyplot as plt
from copy import deepcopy

# PARAMETERS TO CHOOSE ARBITRARILY

alpha_ratios=npr.dirichlet([50,10,10,10,10,50])
total_users=npr.normal(500,10)
graph_weights=npr.random((5,5))

v=np.array([3,3,2,1]).reshape((4,1))
n_items_bought=np.int64(npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v)))
v=np.array([0.8,0.6,0.4,0.2]).reshape((4,1))
conv_rates=npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v))
margins=npr.random((4,5))*20
prices=npr.random((4,5))*10
np.sort(prices)
users= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)

lamb=0.2


def choose_user(users):
    prods=np.nonzeros(users.n_users)
    product=np.random.choice(prods)
    return product

pricing_env=env(prices)

agl=ucb(4,5)

#computation of the regret (pasted from ucb practice session)
N_exp=100
opt=np.max(margins)
cumulative_regret=[]
R=[]
for i in range (N_exp):
    users1=deepcopy(users)
    while users1!=0
        price_pulled=agl.act() 
        j=choose_users(users1)
        users1[j]-=1
        sim = Simulator(prices[[0,1,2,3,4,5],prices_pulled],margins,lamb)
        rew=website_simulation(sim,j)
        agl.update(price_pulled,rew)
        instant_regret=(opt-rew)
    cumulative_regret=np.cumsum(instant_regret)
R.append(cumulative_regret)
mean_R=np.mean(R,axis=0)
std_dev=np.std(R,axis=0)/np.sqrt(N_exp)
plt.plot(mean_R)
best_indices=agl.act()
print('The best prices for each product are:',prices[[0,1,2,3,4,5],best_indices])
