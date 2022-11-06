from simulator import *
from users import *
from website_simulation import website_simulation

# DEFINE THE SIMULATOR
max_margin = 1 # confrontable with the conv_factor
different_value_of_prices = 4
numbers_of_products = 5
margins = npr.random((different_value_of_prices,numbers_of_products)) * max_margin
prices = (npr.rand(numbers_of_products,1)*different_value_of_prices).astype(int).reshape(numbers_of_products)

lamb = 0.2 # LAMBDA
sim = Simulator(prices,margins,lamb)

# EXPLANATION OF THE DISTRIBUTIONS
"""
    Firstly we set 3 constant
        - max_margin: the maximum value of the margin value (not sure)
        - different_value_of_prices: we have 4 values of prices for every product
        - numbers_of_products: the website sell 5 different produts
    
    Margin: type matrix [4,5], were on the 1st dimension we have the different prices per product
            and on the 2nd dimension we have the different product
            Thn we select randomically [maybe normally or with another type of distribution can be better]
            the margin value considering the max_margin constant
    Prices: tpye array [5,1], were we have stored the index of the selected margin for the day
        i think that a random distribution for choosing which between the already generated margin to use is correct.
"""


# DEFINE 3 CLASS OF USERS
# [assume that there are 2 binary features that define 3 different user classes.]
# ???? [WHICH ARE THE TWO BINARY FEATURE THAT DEFINE 3 CLASS OF USERS] ?????
# 00 01 10 11 are 4...

users_classes = 3
max_users_per_class = 100
user_per_class_std = 30 # Standard deviation of the users distribution between classes


# 1 Create the demand curves (conversion rates)

# my other idea...
# Considering a normal distribution over conversion rates,
# i'll use random variable to evaluate the mean of the normal distribution over the class of user
# max_mean_conv = 100
# conv_std = 20
# conv_rates_mean = npr.rand(users_classes) * max_mean_conv
# #conv_rates = map(npr.normal,conv_rates_mean, conv_std)
# conv_rates = [npr.normal(x, user_per_class_std) for x in conv_rates_mean]

conv_rates = npr.rand(users_classes,numbers_of_products)

"""
    Conversion rates are randomically choosed. ??? IDEAS?
"""


# 2 Create num of users
total_users = (npr.normal(max_users_per_class,user_per_class_std,users_classes)).astype(int)
# ::int array randomly from 0 to max_users_per_class
"""
distribution option: 
-   Normal [i think can be a correct idea like implemented]
-   Random 
-   ....
"""

# 3 Create alpha ratios
# TODO add dependency with the users class.

max_prod_weight = 50
product_weight = (npr.rand(numbers_of_products)*max_prod_weight).astype(int)
product_weight[0] = max_prod_weight # [The competitor has higher weight !? ]
alpha_ratios = npr.dirichlet(product_weight,3)
# ::6 ratios, the firs for the COMPETITOR webpage ?????
"""
distribution option:  -   dirichlet (forced)
"""

# 4 Create number of products sold
max_product_sold = 2*total_users
product_sold_std = 10
n_items_bought = (npr.normal(max_product_sold.mean(0),product_sold_std,size = (users_classes,different_value_of_prices,numbers_of_products))).astype(int)
# TODO find a distribution between classes!!!????
"""
distribution option:  
    -   Normal??
    -   Random??
"""


# 5 Create graph probabilities
# TODO graph_weights diagonals must be 0
# TODO 2 type of graph
graph_weights=npr.normal(0.3,0.1,(3,5,5))
# the graph has a doimension of 5x5


# Finally create the users classes
users = []

for i in range(users_classes):
    u = Users_group(total_users[i],alpha_ratios[i],graph_weights[i],n_items_bought[i],conv_rates[i])
    users.append(u)

# EXPLANATION OF THE DISTRIBUTIONS
"""
    TODO write that for the user class on the same format as for the simulator
"""


print(users[0].conv_rates, "Conversion factor")
print(np.diag(margins[prices]),"margins")
print("total revenue", website_simulation(sim,users[0]))

# DOUBTS