"""Additional tools."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from typing import Any, Dict, Iterator, List, Text, Union

from enzynet import constants

import csv

csv.field_size_limit(10**7)

_ELEMENT_DELIMITER = ', '
_STRINGS_TO_IGNORE = ['\'', '"', '[', ']']


def _convert_to_target_value(
        value: Text,
        value_type: constants.ValueType
) -> Union[int, Text, List[float], List[int], List[Text]]:
    """Helper function that turns a string value into its intended type."""
    if value_type == constants.ValueType.INT:
        return int(value)
    elif value_type == constants.ValueType.STRING:
        return value
    for string_to_ignore in _STRINGS_TO_IGNORE:
        value = value.replace(string_to_ignore, '')
    values = value.split(_ELEMENT_DELIMITER)
    if value_type == constants.ValueType.LIST_FLOAT:
        return list(map(float, values))
    elif value_type == constants.ValueType.LIST_INT:
        return list(map(int, values))
    elif value_type == constants.ValueType.LIST_STRING:
        return values
    else:
        raise ValueError(f'Enum value {value_type.name} not supported.')


def read_dict(
        path: Text,
        value_type: constants.ValueType = constants.ValueType.STRING
) -> Dict[Any, Union[int, Text, List[float], List[int], List[Text]]]:
    """Reads Python dictionary stored in a csv file."""
    dictionary = {}
    with open(path) as f:
        for key, val in csv.reader(f):
            dictionary[key] = _convert_to_target_value(val, value_type)
    return dictionary


def dict_to_csv(dictionary: Dict[Any, Any], path: Text) -> None:
    """Saves Python dictionary to a csv file."""
    w = csv.writer(open(path, 'w'))
    for key, val in dictionary.items():
        w.writerow([key, val])


def scale_dict(dictionary: Dict[Any, float]) -> Dict[Any, float]:
    """Scales values of a dictionary."""
    maxi = max(map(abs, dictionary.values()))  # Max in absolute value.
    return {k: v/maxi for k, v in dictionary.items()}


def get_class_weights(dictionary: Dict[Text, int],
                      training_enzymes: Iterator[Text],
                      mode: Text) -> Dict[int, float]:
    """Gets class weights for Keras."""
    # Initialization.
    counter = [0 for i in range(constants.N_CLASSES)]

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
