import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one-hot encoding.
    """
    ohe = pd.get_dummies(X)
    return ohe*1

def isReal(X: pd.Series) -> bool:
    """
    Function to check if the input is real or discrete.
    """
    if pd.api.types.is_string_dtype(X):
        return False
    if pd.api.types.is_numeric_dtype(X):
        if pd.api.types.is_float_dtype(X):
            return True
        else:
            return False
    return False

def entropy(y: pd.Series) -> float:
    """
    Function to calculate entropy.
    """
    bins = np.bincount(y)
    probs = bins / y.size
    return - np.sum([p * np.log2(p) for p in probs if p > 0])

def gini_index(y: pd.Series) -> float:
    """
    Function to calculate Gini Index.
    """
    try:
        bins = np.array(list(Counter(y).values()))
        probs = bins / y.size
        return 1 - np.sum([p**2 for p in probs])
    except:
        print(y)
        raise ValueError("Error in Gini Index")

def gini_gain(parent: pd.Series, left: pd.Series, right: pd.Series) -> float:
    """
    Function to calculate Gini Gain.
    """
    weightLeft, weightRight = left.size / parent.size, right.size / parent.size
    return gini_index(parent) - (weightLeft * gini_index(left) + weightRight * gini_index(right))

def mse(y: pd.Series) -> float:
    """
    Function to calculate Mean Squared Error.
    """
    return ((y - y.mean()) ** 2).mean()

def information_gain(parent: pd.Series, left: pd.Series, right: pd.Series) -> float:
    """
    Function to calculate Information Gain.
    """
    if isReal(parent):
        return mse(parent) - (left.size / parent.size) * mse(left) - (right.size / parent.size) * mse(right)
    
    else:
        weightLeft, weightRight = left.size / parent.size, right.size / parent.size
        return entropy(parent) - (weightLeft * entropy(left) + weightRight * entropy(right))
    

def split(X: pd.DataFrame, y: pd.Series, value: float, feature: str) -> tuple:
    """
    Function to split the data.
    """
    tempDF = X.copy()
    tempDF["output"] = y

    left = tempDF[tempDF[feature] <= value]
    right = tempDF[tempDF[feature] > value]

    y_left, y_right = left["output"], right["output"]
    X_left, X_right = left.drop(columns=["output"]), right.drop(columns=["output"])
    return X_left,X_right,y_left, y_right


def bestSplit(X: pd.DataFrame, y: pd.Series, criterion: str) -> tuple:
    """
    Function to find the best split.
    """
    bestScore = -1
    bestFeature, bestThreshold = None, None

    for feature in X.columns:
        unqVal = X[feature].unique()
        possibleSplits = (unqVal[1:] + unqVal[:-1]) / 2
        for threshold in possibleSplits:
            _,_,y_left, y_right = split(X, y, threshold, feature)
            if criterion == "information_gain":
                currentScore = information_gain(y, y_left, y_right)
            else:
                currentScore = gini_gain(y, y_left, y_right)

            if currentScore > bestScore:
                bestScore = currentScore
                bestFeature = feature
                bestThreshold = threshold
    
    return bestFeature, bestThreshold