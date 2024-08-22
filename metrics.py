from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here

    c = 0
    for i in range(y_hat.size):
        if y_hat[i] == y[i]:
            c += 1
    return c/y_hat.size




def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    tp = 0
    fp = 0
    for i in range(y_hat.size):
        if y_hat[i] == cls and y[i] == cls:
            tp += 1
        elif y_hat[i] == cls and y[i] != cls:
            fp += 1

    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0



def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    tp = 0
    fn = 0
    for i in range(y_hat.size):
        if y_hat[i] == cls and y[i] == cls:
            tp += 1
        elif y_hat[i] != cls and y[i] != cls:
            fn += 1
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0




def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    y = y.fillna(0)
    y_hat = y_hat.fillna(0)
    sqdiff = (y-y_hat)**2
    mean = sqdiff.mean()
    return np.sqrt(mean)





def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """

    absdiff = (y_hat - y).abs()
    return absdiff.mean()
