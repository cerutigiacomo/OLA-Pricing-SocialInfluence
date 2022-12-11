from simulator import *
from users import *
from website_simulation import *
from plotting.plot_distributions import *
from resources.define_distribution import *
import json

debug_print_distribution = False
f = open('../resources/environment.json')
data = json.load(f)


def simple_run():
    reward, a, b, c = website_simulation(sim, users)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("total revenue ", reward)

    plot_reward(reward)


# DEFINE THE SIMULATOR
prices, margins, secondary, today = simulator_distribution()
lamb = data["product"]["lambda"]  # LAMBDA
sim = Simulator(prices, margins, lamb, secondary, [today for _ in range(5)])

# DEFINE 3 CLASS OF USERS
classes_idx = [i for i in range(users_classes)]
users = get_users(classes_idx)

# Plot distributions
if debug_print_distribution:
    plot_simulator(sim)
    plot_users(users, classes_idx)

# RUN the simulation
days = data["simulator"]["days"]
plot_reward(simulate_multiple_days(sim, users, classes_idx))
