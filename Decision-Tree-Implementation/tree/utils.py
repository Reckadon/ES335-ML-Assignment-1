import numpy as np
import pandas as pd


def entropy(series: pd.Series) -> float:
    """
    Function to calculate entropy
    """
    p = series.value_counts(normalize=True)
    return -np.sum(p * np.log2(p))

def giniIndex(series: pd.Series) -> float:
    """
    Function to calculate Gini Index
    """
    p = series.value_counts(normalize=True)
    return 1 - np.sum(p**2)

def MSE(series: pd.Series) -> float:
    """
    Function to calculate Mean Squared Error
    """
    return np.mean((series - series.mean())**2)

def informationGain(parent: pd.Series, left: pd.Series, right: pd.Series, criterion: str) -> float:
    """
    Function to calculate information gain
    """
    if parent.shape[0] == 0:
        return 0
    if criterion == 'entropy':
        return entropy(parent) - (left.shape[0] / parent.shape[0]) * entropy(left) - (right.shape[0] / parent.shape[0]) * entropy(right)
    elif criterion == 'gini_index':
        return giniIndex(parent) - (left.shape[0] / parent.shape[0]) * giniIndex(left) - (right.shape[0] / parent.shape[0]) * giniIndex(right)
    elif criterion == 'mse':
        return MSE(parent) - (left.shape[0] / parent.shape[0]) * MSE(left) - (right.shape[0] / parent.shape[0]) * MSE(right)
    else:
        raise ValueError("Criterion not supported: {}".format(criterion))
def splitData(X: pd.DataFrame, y: pd.Series, attribute: str, val: float, IOtype) -> tuple:
    """
    Function to split the data
    """
    if IOtype[0] == 'real':
        xLeft = X[X[attribute] <= val]
        xRight = X[X[attribute] > val]
    else:
        xLeft = X[X[attribute] == val]
        xRight = X[X[attribute] != val]
    if IOtype[1] == 'real':
        yLeft = y[X[attribute] <= val]
        yRight = y[X[attribute] > val]
    else:
        yLeft = y[X[attribute] == val]
        yRight = y[X[attribute] != val]
    return xLeft, xRight, yLeft, yRight

def bestSplit(X: pd.DataFrame, y: pd.Series, criterion: str, IOtype) -> tuple:
    """
    Function to find the best split
    """
    best_gain = -1
    best_attribute = None
    best_val = None

    if criterion == 'information_gain':
        if IOtype[1] == 'real':
            criteria = 'mse'
        else:
            criteria = 'entropy'
    else:
        criteria = 'gini_index'

    for attribute in X.columns:
        for val in X[attribute].unique():
            if IOtype[1] == 'real':
                left = y[X[attribute] <= val]
                right = y[X[attribute] > val]
            else:
                left = y[X[attribute] == val]
                right = y[X[attribute] != val]
            gain = informationGain(y, left, right, criteria)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                best_val = val
    return best_attribute, best_val

