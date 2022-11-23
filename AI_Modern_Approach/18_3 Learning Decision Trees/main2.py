import math
import random

import numpy as np
from numpy import ndarray

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
    for attribute in attributes:
        attribute_values[attribute] = np.unique(examples[:, np.where(attributes == attribute)[0][0]])
    print(decision_tree_learning(examples, attributes, np.empty(shape=examples.shape, dtype=examples.dtype)))
    print("done")


class Tree:
    def __init__(self, value=None, attribute="LEAF"):
        self.value = value
        self.attribute = attribute
        self.subtrees = {}

    def add_branch(self, attribute, subtree):
        self.subtrees[attribute] = subtree

    def __repr__(self):
        if self.attribute == "LEAF":
            return f"{self.value}"
        else:
            return f"{self.attribute} {self.subtrees}"


def plurality_value(examples: ndarray):
    # return the highest count of the last column of examples, or a random one
    values, counts = np.unique(examples[:, -1], return_counts=True)
    return np.random.choice(values[counts == counts.max()])


def entropy(q: float) -> float:
    if q == 0.0 or q == 1.0:
        return 0.0
    return -1 * (q * math.log2(q) + (1 - q) * math.log2(1 - q))


def max_importance(attributes, examples) -> str:
    positive_count = np.count_nonzero(examples[:, -1] == 'True')
    overall_entropy = entropy(positive_count / (len(examples[:, -1])))
    importance = {}
    for attribute in attributes:
        importance[attribute] = overall_entropy
        attribute_column = examples[:, np.where(attributes == attribute)[0][0]]
        for attribute_value in attribute_values[attribute]:
            goal_for_attribute_value = examples[np.where(attribute_column == attribute_value)[0], -1]
            if len(goal_for_attribute_value) == 0:
                continue
            att_val_pos = np.count_nonzero(goal_for_attribute_value == 'True')
            importance[attribute] -= len(goal_for_attribute_value) / len(examples[:, -1]) * \
                                     entropy(att_val_pos / len(goal_for_attribute_value))
    highest_importance = 0.0
    importance_list = []
    for key, value in importance.items():
        if value > highest_importance:
            highest_importance = value
            importance_list.clear()
        if value == highest_importance:
            importance_list.append(key)
    return random.choice(importance_list)


def decision_tree_learning(examples: ndarray, attributes: ndarray, parent_examples: ndarray) -> Tree:
    if np.size(examples) == 0:
        return Tree(value=plurality_value(parent_examples))
    elif np.all(examples[0, -1] == examples[:, -1]):
        return Tree(value=examples[0, -1])
    elif np.size(attributes) == 0:
        return Tree(value=plurality_value(examples))
    else:
        attribute = max_importance(attributes, examples)
        tree = Tree(attribute=attribute)
        for value in attribute_values[attribute]:
            print(
                f"Sending {attribute} examples to child that match {value}")
            examples_for_value = np.delete(
                examples[np.where(examples[:, np.where(attributes == attribute)[0]] == value)[0], :], np.where(
                    attributes == attribute)[0], axis=1)
            subtree = decision_tree_learning(examples_for_value,
                                             np.delete(attributes, np.where(attributes == attribute)[0]), examples)
            tree.add_branch(value, subtree)
        return tree


if __name__ == "__main__":
    run()
    print("extra done")
