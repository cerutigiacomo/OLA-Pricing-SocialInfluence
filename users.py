class Users_group:
    # This class defines a single group of users that is meant to visit the ecommerce website
    # Pass parameters that correspond to a specific class of users
    def __init__(self, n_users, alpha_ratios, graph_weights, n_items_bought, conv_rates, features) -> None:
        # number of daily users
        self.total_users = int(n_users)
        # 𝛼 ratios
        self.alpha = alpha_ratios
        # class features
        self.features = features
        # graph probabilities
        self.graph_weights = graph_weights
        # number of products sold
        self.n_items_bought = n_items_bought
        # the demand curves of the 5 products
        self.conv_rates = conv_rates
