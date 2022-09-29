from copy import deepcopy
import numpy.random as npr
import numpy as np


class Simulator():
    def __init__(self,prices,margins,lamb) -> None:
        # self.prices are the price levels set for each product
        self.prices=prices
        # margins matrix associated for each product and each price point
        self.margins=margins
        self.lamb=lamb
        self.visited_primaries = []
        
    # def website_simulation(self,user_class):
    # # THIS METHOD IS TO BE REWRITTEN FOR STEP 3 -7
    # # This method simulates users visiting the ecommerce website
    # # argument is an User class instance
    # # returns total rewards for all five products

    #     total_rewards=np.zeros(5,np.float16)

    #     for j in range(len(user_class.n_users)-1):
            
    #         for n in range(round(user_class.n_users[j+1])):
    #             self.visited_primaries=[]
    #             rewards=self.simulation(j,user_class)
    #             total_rewards += rewards

    #         print(round(user_class.n_users[j+1]),"users landing on product", j+1 ,total_rewards)   
        
    #     return total_rewards

    def simulation(self,j,user_class):
        # This recurisve method simulates one user landing on a webpage of one product. 
        # Rewards depends on conversion rates, price point, number of items bought, and margins.
        # After adding to the rewards the money for the primary product, 
        # this method looks into the graph weights to give the most probable
        # secondary products and calls itself recursively to add the rewards of the next primary.

        
        # Compute reward for buying the primary
        rewards = np.zeros(5,np.float16)
        conversion_factor=user_class.conv_rates[self.prices[j]][j]>npr.random()
        rewards[j] = self.margins[self.prices[j]][j]*user_class.n_items_bought[self.prices[j]][j]*conversion_factor
        self.visited_primaries.append(j)

        arr = deepcopy(user_class.graph_weights)[j]
        arr[self.visited_primaries]=0.0

        # Select 2 secondaries with highest observation rates and visit them if weights are positive.
        # If they are null, means they have already been visited.
        first_secondary = np.argmax(arr)
        if arr[first_secondary]>npr.random():
            # print("going to first secondary",first_secondary,"from prim",j)
            rewards +=self.simulation(first_secondary,user_class)
        arr[self.visited_primaries]=0.0
        arr1=deepcopy(arr)
        arr1[first_secondary]=0.0
        second_secondary = np.argmax(arr1)
        if arr[second_secondary]*self.lamb>npr.random():
            # print("going to second secondary",second_secondary,"from prim",j)
            rewards +=self.simulation(second_secondary,user_class)

        # Returns the rewards of that user associated to products bought
        return rewards






        

