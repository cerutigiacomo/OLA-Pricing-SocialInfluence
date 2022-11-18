import numpy as np
import pandas as pd
from scipy.stats import betabinom
from scipy.stats import geom
from math import floor
import matplotlib.pyplot as plt

N_PRODUCTS = 5
N_PRICES = 4

###
###
### does this make sense ? maybe not
items = np.array([20,15,10,5,3,1])
### this is the n items bought corresponding to the index drawn by a random variable
### price are increasing levels so it is coherent to have higher numbers  of bought (lower price) to lower numbers (higher price)
###

n=items.shape[0]-1
a=1
b=N_PRICES

fig, ax = plt.subplots(2,2)
print("Drawing random variable to model number of items bought")

# BetaBinomial
# n fixed to the number of items which index will be drawn
# alpha and beta parameters define the distribution (1,4); (2,3); (3,2); (4,1);

x = range(30)
for i in range(N_PRICES):
    rv = betabinom(n, a, b)
    ax[floor(i/2)][i%2].vlines(x, 0, rv.pmf(x), linestyles='-',lw=1)
    ax[floor(i/2)][i%2].legend(loc='best', frameon=False)
    ax[floor(i / 2)][i % 2].set_title('pmf for price '+str(a))
    a += 1
    b -= 1

txt = "x-axis : index for number of items sold, decreasing with price settings \n" \
      "y-axis : probability to draw a specific number_of_item_sold index"

fig.text(.5, .05, txt, ha='center')
fig.tight_layout(pad=3.9)
fig.suptitle("BetaBinomial PMF - number of products bought for different price settings", fontsize=10)

plt.legend()
plt.show()


########
########
########


n=items.shape[0]-1
a=1
b=N_PRICES

fig,ax = plt.subplots()

n_boughts = np.empty(shape=(N_PRICES,N_PRODUCTS))
for price_setting in range(N_PRICES):
    print(a,b)
    index_of_bought_n = betabinom.rvs(n, a, b, size=N_PRODUCTS)
    print(index_of_bought_n)
    n_boughts[price_setting,:] = np.array([items[index] for index in index_of_bought_n]).T
    a += 1
    b -= 1

print("\nShape of random drawings : ",n_boughts.shape)

row_labels = ['$1', '$2', '$3', '$4']
column_labels = ['A', 'B', 'C', 'D','E']
df = pd.DataFrame(n_boughts, columns=column_labels, index=row_labels)
print(df)
print("\n\n\n")


for product_id in range(N_PRODUCTS):
    col = n_boughts[:,product_id]
    print(col)
    # col refers to an array of size N_PRICES containing the number of sell for product i
    label = "prod"+str(product_id)
    ax.plot(range(N_PRICES),col,'-o',lw=0.5,label=label)
    ax.set_xticks(range(N_PRICES))
    ax.set_yticks(items)
    ax.set_xlabel("price setting")
    ax.set_ylabel("n. items bought")

plt.legend()
plt.show()