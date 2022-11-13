from simulator import *
from users import *
from website_simulation import website_simulation

# DEFINE THE SIMULATOR
max_margin = 1
different_value_of_prices = 4
numbers_of_products = 5
margins = npr.random((different_value_of_prices, numbers_of_products)) * max_margin
prices = (npr.rand(numbers_of_products, 1) * different_value_of_prices).astype(int).reshape(numbers_of_products)

# Define the pair and the order of the secondary products to display

# Init the array
secondary = np.zeros((numbers_of_products, 2))
all_contained = False
# Every product should have different secondary product (different from himself)
# Every product has to be reachable, so all the product has to be contained in the secondary array [????]
while not all_contained:
    for i in range(numbers_of_products):
        secondary[i] = npr.choice(numbers_of_products, 2, replace=False)
        while np.isin(secondary[i], i).any():
            secondary[i] = npr.choice(numbers_of_products, 2, replace=False)
    values = np.isin(secondary, [0, 1, 2, 3, 4])
    all_contained = values.all()

lamb = 0.2  # LAMBDA
sim = Simulator(prices, margins, lamb, secondary)

# EXPLANATION OF THE DISTRIBUTIONS
"""
    Firstly we set 3 constant
        - max_margin: the maximum value of the margin value (not sure)
        - different_value_of_prices: we have 4 values of prices for every product
        - numbers_of_products: the website sell 5 different products
    
    Margin: type matrix [4,5], were on the 1st dimension we have the different prices per product
            and on the 2nd dimension we have the different product
            Thn we random select [maybe normally or with another type of distribution can be better]
            the margin value considering the max_margin constant
    Prices: type array [5,1], were we have stored the index of the selected margin for the day
        i think that a random distribution for choosing which between the already generated margin to use is correct.
"""


# DEFINE 3 CLASS OF USERS
# [assume that there are 2 binary features that define 3 different user classes.]
# ???? [WHICH ARE THE TWO BINARY FEATURE THAT DEFINE 3 CLASS OF USERS] ?????
# 00 01 10 11 are 4...

users_classes = 3


# 1 Create the demand curves (conversion rates)
conv_rates = np.zeros((users_classes, numbers_of_products))
max_demand = np.zeros(users_classes)
min_conv_rates = 0.3

for i in range(users_classes):
    # Define a maximum value of conversion rates for every user class
    max_demand[i] = npr.random()
    conv_rates[i] = npr.uniform(min_conv_rates, max_demand[i], numbers_of_products)

"""
    Conversion rates are chosen uniform between 0 and a max[i]
    max_demand[i] is the maximum value of conversion rate of a single user class, random chosen
"""


# 2 Create num of users
# max_users_per_class = 100
# user_per_class_std = 30 # Standard deviation of the user's distribution between classes
# total_users = (npr.normal(max_users_per_class,user_per_class_std,users_classes)).astype(int)

total_users = np.zeros(users_classes)
for i in range(users_classes):
    # TODO not sure??
    max_users_per_class = 300
    user_per_class_std = 30  # Standard deviation of the user's distribution between classes
    total_users[i] = (npr.normal(npr.random()*max_users_per_class,
                                 npr.random() * user_per_class_std, 1)).astype(int)

# ::int array randomly from 0 to max_users_per_class
"""
distribution option: 
-   Normal [i think can be a correct idea like implemented]
-   Random 
-   ....
the mean is a random variable from 0 to 300
the std is a random variable from 0 to 30
"""

# 3 Create alpha ratios
# TODO add dependency with the users class.

product_weight = (npr.uniform(0.4, 1, numbers_of_products+1))
product_weight[0] = 0.5  # [The competitor has higher weight ! ]
alpha_ratios = npr.dirichlet(product_weight, users_classes)
# 6 ratios, the first is for the COMPETITOR webpage
"""
distribution option:  -   dirichlet (forced)
distribution weights are chosen random.
"""

# 4 Create number of products sold
# TODO change!!!
n_items_bought = np.zeros((users_classes, different_value_of_prices, numbers_of_products))
for i in range(users_classes):
    # TODO not sure??
    max_item_bought = 5
    n_items_bought_std = 2  # Standard deviation of the user's distribution between classes
    n_items_bought[i] = (npr.normal(npr.random()*max_item_bought,
                                    npr.random() * n_items_bought_std,
                                    (different_value_of_prices, numbers_of_products))).astype(int)

# TODO find a distribution between classes!!!????
"""
distribution option:  
    -   Normal??
    -   Random??
"""


# 5 Create graph probabilities
# OLD_TODO graph_weights diagonals must be 0 -> we do not care since the secondary product cannot be the first one
# form the previous definition
# old #graph_weights=npr.normal(0.3,0.1,(3,5,5))

# FULLY CONNECTED
graph_weights = np.zeros((users_classes, numbers_of_products, numbers_of_products))
max_demand = np.zeros(users_classes)
for i in range(users_classes):
    # Define a maximum value of graph_weight for every user class
    max_demand[i] = npr.uniform(0.4, 1)
    graph_weights[i] = npr.uniform(0, max_demand[i], (numbers_of_products, numbers_of_products))

# second type of graph set to 0 only certain values
# NOT FULLY CONNECTED
graph_weights_not_fully_connected = np.copy(graph_weights)
for i in range(numbers_of_products):
    how_many_set_to_zero = int(round(numbers_of_products*npr.random()))
    set_to_zero = (npr.choice(numbers_of_products, how_many_set_to_zero, replace=False)).astype(int)
    for j in set_to_zero:
        j = int(j)
        # TODO Equally between all the users classes ???
        graph_weights_not_fully_connected[:, i, j] = 0

# Finally, create the users classes
users = []

for i in range(users_classes):
    u = Users_group(total_users[i], alpha_ratios[i], graph_weights[i], n_items_bought[i], conv_rates[i])
    users.append(u)

# EXPLANATION OF THE DISTRIBUTIONS
"""
    TODO write that for the user class on the same format as for the simulator
"""

# Print simulator distributions
print("SIMULATOR DISTRIBUTION", end="\n\n")
print("prices\n", prices)
print("margins\n", margins)
print("lambda\n", lamb)
print("secondary\n", secondary)

# Print Users distributions
print("\n\n\nUSER DISTRIBUTION", end="\n\n")
print("Total users\n", total_users)
print("Alpha ratios\n", alpha_ratios)
print("Graph weights\n", graph_weights)
print("N items bought\n",n_items_bought)
print("Conversion rates\n", conv_rates)

reward = np.zeros(numbers_of_products)
for i in range(users_classes):
    reward += website_simulation(sim, users[i])
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("total revenue ", reward)
