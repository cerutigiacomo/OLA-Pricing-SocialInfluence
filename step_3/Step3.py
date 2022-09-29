import numpy.random as npr
from users import *
import matplotlib.pyplot as plt
from copy import deepcopy

#Here is an adaptation of the UCB code, inspired by the code of the practice session
# The main difference is that we have an optimization for 5 different items instead of 1
# So all the matrices have 2 dimensions (product and price) instead of one.
# The arms of the UCB algorithm are the prices of each item, that can take 4 different values.
          
class env:
    def __init__(self,probs,prices):
        self.prices=prices

class learner:
    def __init__(self,n_products,n_prices):
        self.n_products=n_products
        self.n_prices=n_prices
        self.collected_rewards=[]
        self.rewards=[[]for i in range (n_prices)]*n_products

    def act(self):
        pass

    def update(self,prices_pulled, reward):
        for k in range(5):
            self.rewards[k][prices_pulled[k]].append(reward)

class ucb(learner):
    def __init__(self, n_products,n_prices):
        super().__init__(n_products,n_prices)
        self.means=np.zeros(n_products,n_prices)
        self.widths=np.array([np.inf for _ in range(n_prices)]*n_products)
            
    def act(self):
        idx=np.argmax(self.means+self.widths,axis=1)
        return idx

    def update(self,prices_pulled,reward):
            #price_pulled is a list of chosen prices for each product
            #as we have 5 products here it is a list of 5 prices
                super().update(prices_pulled,reward)
                for j in range (n_products):
                    self.means=np.mean(self.rewards[j][price_pulled[j]])
                    for idx in range (self.n_prices):
                        n=len(self.rewards[j][idx])
                        if n>0:
                            self.widths[j][idx]=np.sqrt(2*np.log(self.t)/n)
                        else: 
                            self.width[j][idx]=np.inf
                


