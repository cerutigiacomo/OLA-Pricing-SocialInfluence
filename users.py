import numpy as np
import numpy.random as npr

class Users_group():
# This class defines a single group of users that is meant to visit the ecommerce website
# Pass parameters that correspond to a specific class of users
# Class values are all random variables arbitrarily chosen
    def __init__(self,n_users,alpha_ratios,graph_weights,n_items_bought,conv_rates) -> None:
        self.total_users=n_users
        # Normal distribution for number of total users of the group
        self.n_users=alpha_ratios*self.total_users
        # Alpha ratios is dirichlet distribution
        # First element is number of users that visit competitor website, 
        # and the rest is the number of users starting to browse the website on product 'i'
        self.graph_weights=graph_weights
        # Weighted adjacency matrix of the graph, size 5x5
        self.n_items_bought=n_items_bought
        # Normal rv for number of items bought
        self.conv_rates=conv_rates


if __name__  == "__main__":
# example

    alpha_ratios=npr.dirichlet([50,10,10,10,10,50])
    total_users=npr.normal(500,10)
    graph_weights=npr.random((5,5))

    v=np.array([3,3,2,1]).reshape((4,1))
    n_items_bought=np.int64(npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v)))
    v=np.array([0.8,0.6,0.4,0.2]).reshape((4,1))
    conv_rates=npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v))

    users= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
    print(users.graph_weights)


