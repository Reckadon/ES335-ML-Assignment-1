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



def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real (continuous) or discrete (categorical) values.

    Parameters:
    y (pd.Series): The series to check.

    Returns:
    bool: True if the series contains real (continuous) values, False if it contains discrete (categorical) values.
    """

    # # Check if the series is of float type or contains any non-integer values
    # if pd.api.types.is_float_dtype(y):
    #     return True
    # if pd.api.types.is_numeric_dtype(y) and not all(y == y.astype(int)):
    #     return True
    #
    # # If the series is of object or category type, it is likely discrete
    # if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
    #     return False

    # Check if the series has a small number of unique values compared to its length
    if y.nunique() < 0.5 * len(y):  # Adjust the threshold as necessary
        return False

    # Default to discrete if none of the above apply
    return True


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

def MSE(Y: pd.Series) -> float:
    try:
        return ((Y - Y.mean()) ** 2).mean()
    except:
        return entropy(Y)


def information_gain(parent: pd.Series, left: pd.Series, right: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    ig = 0
    if criterion == 'entropy':
        parent_entropy = entropy(parent)
        weighted_entropy = (len(left) / len(parent)) * entropy(left) + (len(right) / len(parent)) * entropy(right)
        ig = parent_entropy - weighted_entropy

    elif criterion == 'gini':
        parent_gini = gini_index(parent)
        weighted_gini = (len(left) / len(parent)) * gini_index(left) + (len(right) / len(parent)) * gini_index(right)
        ig = parent_gini - weighted_gini

    elif criterion == 'MSE':
        parent_mse = MSE(parent)
        weighted_mse = (len(left) / len(parent)) * MSE(left) + (len(right) / len(parent)) * MSE(right)
        ig = parent_mse - weighted_mse

    else:
        raise ValueError(f'Criterion should be either entropy, gini or MSE: {criterion}')
    return ig


def split_data(x: pd.Series, y: pd.Series, threshold=None):
    if threshold is None:  # for discreete
        uval = x.unique()
        split = []
        for value in uval:
            left_mask = (x == value)
            right_mask = (x != value)
            left_y = y[left_mask]
            right_y = y[right_mask]
            split.append((value, left_y, right_y))
        return split
    else:  # Continuous split
        left_mask = (x <= threshold)
        right_mask = (x > threshold)
        left_y = y[left_mask]
        right_y = y[right_mask]
        return [(threshold, left_y, right_y)]


def opt_split_attribute(x: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_attribute = None
    best_info_gain = -float('inf')
    best_gini = float('inf')
    best_split_value = None

    for feature in features:
        x_feature = x[feature]
        print(x_feature)
        if x_feature.dtype in [np.float64, np.int64]:  # Real
            unique_values = np.unique(x_feature)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                splits = split_data(x_feature, y, threshold)
                for split_value, left_y, right_y in splits:
                    if len(left_y) > 0 and len(right_y) > 0:
                        if criterion == 'information_gain' or criterion == 'MSE' or criterion == 'entropy':
                            ig = information_gain(y, left_y, right_y, 'MSE')
                            print(f"QWERTY {ig}")
                        elif criterion == 'gini_index':
                            ig = information_gain(y, left_y, right_y, 'gini')
                        else:
                            raise ValueError(f'Unsupported criterion: {criterion}')

                        if ig > best_info_gain:
                            best_info_gain = ig
                            best_attribute = feature
                            best_split_value = split_value

        else:  # Discrete
            splits = split_data(x_feature, y)
            for split_value, left_y, right_y in splits:
                if len(left_y) > 0 and len(right_y) > 0:
                    if criterion == 'information_gain' or criterion == 'entropy' or criterion == 'MSE':
                        ig = information_gain(y, left_y, right_y, 'entropy')
                    elif criterion == 'gini_index':
                        ig = information_gain(y, left_y, right_y, 'gini')
                    # else:
                    #     raise ValueError(f'Unsupported criterion: {criterion}')

                    if criterion == 'information gain' and ig > best_info_gain:
                        best_info_gain = ig
                        best_attribute = feature
                        best_split_value = split_value
                    elif criterion == 'gini index' and ig < best_gini:
                        best_gini = ig
                        best_attribute = feature
                        best_split_value = split_value

    return best_attribute, best_split_value


# Example usage:
# best_attribute, best_split_value = opt_split_attribute(x, y, criterion='information gain', features=x.columns)


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


def splitdataframe(x: pd.DataFrame, y: pd.Series, attribute, value):


# """
#     Funtion to split the data according to an attribute.
#     If needed you can split this function into 2, one for discrete and one for real valued features.
#     You can also change the parameters of this function according to your implementation.
#
#     attribute: attribute/feature to split upon
#     value: value of that attribute to split upon
#
#     return: splitted data(Input and output)
#     """
    if check_ifreal(x[attribute]):  # Real
        # Split based on threshold
        left_mask = x[attribute] <= value
        right_mask = x[attribute] > value

    else:  # Discrete
        # Split based on specific value
        left_mask = x[attribute] == value
        right_mask = x[attribute] != value

    left_x = x[left_mask]
    left_y = y[left_mask]
    right_x = x[right_mask]
    right_y = y[right_mask]

    return left_x, left_y, right_x, right_y
#
#     # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
#     isReal = check_ifreal(X[attribute])
#     if isReal:
#         xLeft = X[X[attribute] <= value].reset_index(drop=True)
#         xRight = X[X[attribute] > value].reset_index(drop=True)
#         yLeft = y[X[attribute] <= value].reset_index(drop=True)
#         yRight = y[X[attribute] > value].reset_index(drop=True)
#     else:
#         xLeft = X[X[attribute] == value].reset_index(drop=True)
#         xRight = X[X[attribute] != value].reset_index(drop=True)
#         yLeft = y[X[attribute] == value].reset_index(drop=True)
#         yRight = y[X[attribute] != value].reset_index(drop=True)
#
#     return xLeft, yLeft, xRight, yRight

# attribute,value = opt_split_attribute(x, y, 'entropy', features)
#
# print(split_data(x, y, attribute, value))

# def findSplitValue(X: pd.DataFrame, y: pd.Series, attribute, criterion):
#     """
#     Function to find the optimal value to split a real valued attribute upon
#     """
#
#     attributeToSplit = X[attribute]
#     isReal = check_ifreal(attributeToSplit)
#     score = 0
#
#     if not isReal:
#         sortedAttribute = attributeToSplit.sort_values()
#         uniqueValues = sortedAttribute.unique()
#
#         bestValue = None
#         bestScore = -float('inf')
#
#         possibleSplitPoints = uniqueValues
#
#         for splitPoint in possibleSplitPoints:
#             xLeft, yLeft, xRight, yRight = split_data(X, y, attribute, splitPoint)
#             if criterion == 'entropy':
#                 entropyBefore = entropy(y)
#                 entropyAfter = (yLeft.size / y.size) * entropy(yLeft) + (
#                             yRight.size / y.size) * entropy(yRight)
#                 score = entropyBefore - entropyAfter
#
#             elif criterion == 'gini':
#                 giniBefore = gini_index(y)
#                 giniAfter = (yLeft.size / y.size) * gini_index(yLeft) + (
#                             yRight.size / y.size) * gini_index(yRight)
#                 score = giniBefore - giniAfter
#
#             if score > bestScore:
#                 bestValue = score
#                 bestScore = splitPoint
#         return bestValue
#
#     else:
#         sortedAttribute = attributeToSplit.sort_values().unique()
#
#         bestValue = None
#         bestScore = -float('inf')
#
#         possibleSplitPoints = (sortedAttribute[1:] + sortedAttribute[:-1]) / 2
#

#         for splitPoint in possibleSplitPoints:
#             xLeft, yLeft, xRight, yRight = split_data(X, y, attribute, splitPoint)
#             if criterion == 'MSE':
#                 mseBefore = ((y - y.mean()) ** 2).mean()
#                 mseAfter = (yLeft.size / y.size) * ((yLeft - yLeft.mean()) ** 2).mean() + (
#                 yRight.size / y.size) * ((yRight - yRight.mean()) ** 2).mean()
#                 score = mseBefore - mseAfter
#             if score > bestScore:
#                 bestValue = score
#                 bestScore = splitPoint
#
#         return bestValue
# #
# print(findSplitValue(x,y,'A','entropy'))
