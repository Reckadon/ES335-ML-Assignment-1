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



def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real (continuous) or discrete (categorical) values.

    Parameters:
    y (pd.Series): The series to check.

    Returns:
    bool: True if the series contains real (continuous) values, False if it contains discrete (categorical) values.
    """

    # Check if the series is of float type or contains any non-integer values
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_numeric_dtype(y) and not all(y == y.astype(int)):
        return True

    # If the series is of object or category type, it is likely discrete
    if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return False

    # Check if the series has a small number of unique values compared to its length
    if y.nunique() < 0.05 * len(y):  # Adjust the threshold as necessary
        return False

    # Default to discrete if none of the above apply
    return False



def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy of a dataset Y.

    Returns:
    float: The entropy value, ranging from 0 (pure) to 1 (max impurity).
    """
    m = list(Y)
    counter = {}
    for i in m:
        if i in counter:
            counter[i] += 1
        else:
            counter[i] = 1
    entropy = 0
    for count in counter.values():
        p = count / len(m)
        entropy -= p * math.log2(p)
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the Gini Index of a dataset Y.

    Returns:
    float: The Gini Index value, ranging from 0 (pure) to 1 (max impurity).
    """
    m = list(Y)
    counter = {}
    for i in m:
        if i in counter:
            counter[i] += 1
        else:
            counter[i] = 1

    gini_sum = 0
    for count in counter.values():
        p = count / len(m)
        gini_sum += p ** 2
    gini = 1 - gini_sum

    return gini


# , attr: pd.Series



def information_gain(Y: pd.Series, criterion: str) -> float:
        """
        Function to calculate the information gain using criterion (entropy, gini index or MSE)
        """
        try:
            if criterion == 'entropy':
                return entropy(Y)
            elif criterion == 'gini':
                return gini_index(Y)
            elif criterion == 'MSE':
                mean_value = Y.mean()
                mse = ((Y - mean_value) ** 2).mean()
                return mse
        except ValueError:
            print("VE")





def opt_split_attribute(x: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    req_fet = None
    # best_score = float('-inf') if criterion == 'entropy' else float('inf')
    if criterion == 'entropy' or criterion == 'MSE':
        best_score = float('-inf')
    else:
        best_score = float('inf')
    for i in features:
        if criterion == 'entropy' or criterion == 'MSE':
            if check_ifreal(y):
                msebefore = information_gain(y, 'MSE')
                mseafter = 0
                for _,grp in y.groupby(x[i]):
                    mseafter += information_gain(grp, 'MSE') * grp.size/y.size
                score = msebefore - mseafter
            else:
                entropybefore = information_gain(y, 'entropy')
                entropyafter = 0
                for _,grp in y.groupby(x[i]):
                    entropyafter += information_gain(grp, 'entropy') * grp.size/y.size
                score = entropybefore - entropyafter
            if score > best_score:
                best_score = score
                req_fet = i
        elif criterion == 'gini':
            score = 0
            for _,grp in y.groupby(x[i]):
                score += information_gain(grp, 'gini') * grp.size/y.size
            if score < best_score:
                best_score = score
                req_fet = i
    return req_fet





# x = pd.DataFrame({
#         'A': [1, 2, 3, 4],
#         'B': [1, 1, 0, 0]
#     })
# y = pd.Series([1, 1, 0, 0])
# features = pd.Series(['A', 'B'])
#
# print(check_ifreal(y))
#
#
# result = opt_split_attribute(x, y, 'entropy', features)
#
#
# print(result)


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
    isReal = check_ifreal(X[attribute])
    if isReal:
        xLeft = X[X[attribute] <= value].reset_index(drop=True)
        xRight = X[X[attribute] > value].reset_index(drop=True)
        yLeft = y[X[attribute] <= value].reset_index(drop=True)
        yRight = y[X[attribute] > value].reset_index(drop=True)
    else:
        xLeft = X[X[attribute] == value].reset_index(drop=True)
        xRight = X[X[attribute] != value].reset_index(drop=True)
        yLeft = y[X[attribute] == value].reset_index(drop=True)
        yRight = y[X[attribute] != value].reset_index(drop=True)

    return xLeft, yLeft, xRight, yRight

# attribute,value = opt_split_attribute(x, y, 'entropy', features)
#
# print(split_data(x, y, attribute, value))

def findSplitValue(X: pd.DataFrame, y: pd.Series, attribute, criterion):
    """
    Function to find the optimal value to split a real valued attribute upon
    """

    attributeToSplit = X[attribute]
    isReal = check_ifreal(attributeToSplit)
    score = 0

    if not isReal:
        sortedAttribute = attributeToSplit.sort_values()
        uniqueValues = sortedAttribute.unique()

        bestValue = None
        bestScore = -float('inf')

        possibleSplitPoints = uniqueValues

        for splitPoint in possibleSplitPoints:
            xLeft, yLeft, xRight, yRight = split_data(X, y, attribute, splitPoint)
            if criterion == 'entropy':
                entropyBefore = entropy(y)
                entropyAfter = (yLeft.size / y.size) * entropy(yLeft) + (
                            yRight.size / y.size) * entropy(yRight)
                score = entropyBefore - entropyAfter

            elif criterion == 'gini':
                giniBefore = gini_index(y)
                giniAfter = (yLeft.size / y.size) * gini_index(yLeft) + (
                            yRight.size / y.size) * gini_index(yRight)
                score = giniBefore - giniAfter

            if score > bestScore:
                bestValue = score
                bestScore = splitPoint
        return bestValue

    else:
        sortedAttribute = attributeToSplit.sort_values().unique()

        bestValue = None
        bestScore = -float('inf')

        possibleSplitPoints = (sortedAttribute[1:] + sortedAttribute[:-1]) / 2

        for splitPoint in possibleSplitPoints:
            xLeft, yLeft, xRight, yRight = split_data(X, y, attribute, splitPoint)
            if criterion == 'MSE':
                mseBefore = ((y - y.mean()) ** 2).mean()
                mseAfter = (yLeft.size / y.size) * ((yLeft - yLeft.mean()) ** 2).mean() + (
                yRight.size / y.size) * ((yRight - yRight.mean()) ** 2).mean()
                score = mseBefore - mseAfter
            if score > bestScore:
                bestValue = score
                bestScore = splitPoint

        return bestValue
#
# print(findSplitValue(x,y,'A','entropy'))