import numpy.random as npr
import json

f = open('../resources/environment.json')
data = json.load(f)

def get_right_user_class(classes=None):
    """
    Gets the class of the Customer considering the fractions of probabilities on the .json file.
    """

    probs = [data["users"]["classes"][c]['fraction_between_other_classes'] for c in classes]

    class_idx = npr.choice(3, 1, p=probs)
    features = data["users"]["classes"][class_idx[0]]['features']

    return class_idx, features