from simulator import *
from users import *
from website_simulation import website_simulation


alpha_ratios=npr.dirichlet([50,10,10,10,10,50])
total_users=npr.normal(500,10)
graph_weights=npr.random((5,5))

v=np.array([3,3,2,1]).reshape((4,1))
n_items_bought=np.int64(npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v)))
v=np.array([0.8,0.6,0.4,0.2]).reshape((4,1))
conv_rates=npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v))

margins=npr.random((4,5))*20
prices=[1,3,2,2,0]
lamb=0.2

users_A= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
sim = Simulator(prices,margins,lamb)

print(np.diag(users_A.conv_rates[prices]),"conv_rates")
print(np.diag(margins[prices]),"margins")
print(website_simulation(sim,users_A))
