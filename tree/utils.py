"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import math

import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

    pass


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    m = list(y)
    if len(set(m)) / len((m)) > 0.2:
        return True
    return False

    pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    m = list(Y)
    d = {}
    for i in m:
        d[i] = d.get(i,0) + 1

    entropy = 0
    for i in d.values():
        entropy += i * math.log(i, 2)
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    m = list(Y)
    d = {}
    for i in set(m):
        d[i] = m.count(i) / len(m)
    gini = 0
    for i in d.values():
        gini += i ** 2
    return 1 - gini


# , attr: pd.Series

def information_gain(Y: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == 'entropy':
        return entropy(Y)
    elif criterion == 'gini':
        return gini_index(Y)
    elif criterion == 'MSE':
        m = list(Y)
        mean = Y.mean()
        c = 0
        for i in m:
            c += (i - mean) ** 2
        return c


def opt_split_attribute(x: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    reqfeature = ""
    best = np.inf

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    # for i in features:
    #     if check_ifreal(y):
            # sety = x[i].unique()
            # splitscore = []
            # for j in sety:
            #     l = x[x[i] <= j]
            #     r = x[x[i] > j]
            #
            #     score = len(l) * information_gain(l, criterion) / len(y) + len(r) * information_gain(r,
            #                                                                                          criterion) / len(
            #         y)
            #     splitscore.append((j,score))
            # a,b = min(splitscore,key=lambda x: x[1])
        # else:
        #     a = information_gain(y, criterion)
        # if criterion = "gini" and a < best:
        # best = a
        # reqfeature = i
        # elif criterion = "MSE" and a < best:








def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    pass
# a pd l