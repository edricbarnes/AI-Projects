import math

import numpy as np
from numpy import ndarray

import solution
import test

attribute_values = {}


def run():
    examples = np.array([[True, False, False, True, "Some", "$$$", False, True, "French", "0-10", True],
                         [True, False, False, True, "Full", "$", False, False, "Thai", "30-60", False],
                         [False, True, False, False, "Some", "$", False, False, "Burger", "0-10", True],
                         [True, False, True, True, "Full", "$", True, False, "Thai", "10-30", True],
                         [True, False, True, False, "Full", "$$$", False, True, "French", ">60", False],
                         [False, True, False, True, "Some", "$$", True, True, "Italian", "0-10", True],
                         [False, True, False, False, "None", "$", True, False, "Burger", "0-10", False],
                         [False, False, False, True, "Some", "$$", True, True, "Thai", "0-10", True],
                         [False, True, True, False, "Full", "$", True, False, "Burger", ">60", False],
                         [True, True, True, True, "Full", "$$$", False, True, "Italian", "10-30", False],
                         [False, False, False, False, "None", "$", False, False, "Thai", "0-10", False],
                         [True, True, True, True, "Full", "$", False, False, "Burger", "30-60", True]])
    attributes = np.array(
        ["Alternate", "Bar", "Friday", "Hungry", "Patrons", "Price", "Rain", "Reservation", "Type", "Est"])
    for i, value in enumerate(attributes):
        attribute_values[value] = np.unique(examples[:, i])
    print(decision_tree_learning(examples, attributes, np.empty(shape=examples.shape, dtype=examples.dtype)))


class Tree:
    def __init__(self, attribute: str = "LEAF", leaf_value: object = None):
        self.attribute = attribute
        self.subtrees = {}
        self.leaf_value = leaf_value

    def add_branch(self, attribute: str, subtree: 'Tree') -> None:
        self.subtrees[attribute] = subtree

    def __repr__(self):
        if self.attribute == "LEAF":
            return f"{self.leaf_value}"
        else:
            return f"{self.attribute} {self.subtrees}"


def plurality_value(examples: ndarray) -> str:
    values, counts = np.unique(examples, return_counts=True)
    return np.random.choice(values[counts == counts.max()])


def boolean_entropy(q: float) -> float:
    if q == 0.0 or q == 1.0:
        return 0.0
    return -1 * (q * math.log2(q) + (1 - q) * math.log2(1 - q))


# This function only works when the result is a boolean
def calculate_information_gain(examples_slice: ndarray, results: ndarray) -> float:
    total_positive = np.count_nonzero(results == "True")
    total_negative = len(results) - total_positive
    remainder = 0.0
    for value in np.unique(examples_slice):
        attribute_value_results = results[examples_slice == value]
        positive_for_this_attribute_value = np.count_nonzero(attribute_value_results == "True")
        negative_for_this_attribute_value = len(attribute_value_results) - positive_for_this_attribute_value
        remainder += (positive_for_this_attribute_value + negative_for_this_attribute_value) / (
                total_positive + total_negative) * boolean_entropy(positive_for_this_attribute_value / (
                positive_for_this_attribute_value + negative_for_this_attribute_value))
    return boolean_entropy(total_positive / (total_negative + total_positive)) - remainder


def max_importance_index(attributes: ndarray, examples: ndarray) -> int:
    information_gain = np.array(
        [calculate_information_gain(examples[:, index], examples[:, -1]) for index in range(len(attributes))])
    information_gain2 = np.array(
        [solution.DecisionTree.information_gain(examples[:, index], examples[:, -1]) for index in range(len(attributes))])
    if np.all(information_gain == information_gain2):
        print("it works")
    return np.argmax(information_gain).item()


def decision_tree_learning(examples: ndarray, attributes: ndarray, parent_examples: np.ndarray) -> Tree:
    if np.size(examples) == 0:
        return Tree(leaf_value=plurality_value(parent_examples[..., -1]))
    elif np.all(examples[0, -1] == examples[:, -1]):
        return Tree(leaf_value=examples[0, -1])
    elif np.size(attributes) == 0:
        return Tree(leaf_value=plurality_value(examples[..., -1]))
    else:
        attribute_column = max_importance_index(attributes, examples)
        tree = Tree(attribute=attributes[attribute_column])
        for attribute_value in attribute_values[attributes[attribute_column]]:
            print(
                f"Sending {attributes[attribute_column]} examples to child that match {attribute_value}")
            attribute_examples = np.delete(examples[examples[:, attribute_column] == attribute_value, :],
                                           attribute_column, axis=1)
            subtree = decision_tree_learning(attribute_examples, np.delete(attributes, attribute_column), examples)
            tree.add_branch(attribute_value, subtree)
        return tree


if __name__ == "__main__":
    run()
