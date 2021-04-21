"""Additional tools."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

import csv
import threading


csv.field_size_limit(10**7)


def read_dict(path):
    """Reads Python dictionary stored in a csv file."""
    dictionary = {}
    for key, val in csv.reader(open(path)):
        dictionary[key] = val
    return dictionary


def dict_to_csv(dictionary, path):
    """Saves Python dictionary to a csv file."""
    w = csv.writer(open(path, 'w'))
    for key, val in dictionary.items():
        w.writerow([key, val])


def scale_dict(dictionary):
    """Scales values of a dictionary."""
    maxi = max(map(abs, dictionary.values()))  # Max in absolute value.
    return {k: v/maxi for k, v in dictionary.items()}


def get_class_weights(dictionary, training_enzymes, mode):
    """Gets class weights for Keras."""
    # Initialization.
    counter = [0 for i in range(6)]

    # Count classes.
    for enzyme in training_enzymes:
        counter[int(dictionary[enzyme])-1] += 1
    majority = max(counter)

    # Make dictionary.
    class_weights = {i: float(majority/count) for i, count in enumerate(counter)}

    # Value according to mode.
    if mode == 'unbalanced':
        for key in class_weights:
            class_weights[key] = 1
    elif mode == 'balanced':
        pass
    elif mode == 'mean_1_balanced':
        for key in class_weights:
            class_weights[key] = (1+class_weights[key])/2

    return class_weights
