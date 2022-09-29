import numpy.random as npr
import numpy as np
from simulator import *
from users import *
from Environment import *
from TS import *
import matplotlib.pyplot as plt

# Optimization with uncertain conversion rates, 𝛼 ratios, and number of items sold per
# product. Do the same of Step 3 when also the alpha ratios and the number of items sold per
# product are uncertain. Develop the algorithms by Python and evaluate their performance
# when applied to your simulator.

# UNCERTAIN values
alpha_ratios=npr.dirichlet([50,10,10,10,10,50])
n_items_bought=np.int64(npr.normal(1,0.2,(4,5)))
conv_rates=npr.normal(1,0.2,(4,5))


# ESTIMABLE values
total_users=npr.normal(500,10)
graph_weights=npr.random((5,5))
margins=npr.random((4,5))*20
lamb=0.2

users= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)


# Environment
pricing_env=Environment(conv_rates, margins)
alg = TS_Learner()


# Computation of the regret (pasted from ucb practice session)
T=1000
N_exp=100
opt=np.max(margins)
cumulative_regret=[]
R=[]
for i in range (N_exp):
    users1=deepcopy(users)
    while users1!=0:
        price_pulled=alg.act()
        rew=pricing_env.round(price_pulled)
        alg.update(price_pulled,rew,users1)
        instant_regret=(opt-rew)
    cumulative_regret=np.cumsum(instant_regret)

R.append(cumulative_regret)
mean_R=np.mean(R,axis=0)
std_dev=np.std(R,axis=0)/np.sqrt(N_exp)
plt.plot(mean_R)


