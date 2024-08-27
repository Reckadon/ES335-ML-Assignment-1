from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate accuracy.
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y_hat.size > 0, "Input series must not be empty."
    
    c = (y_hat == y).sum()
    return c / y_hat.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate precision.
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y_hat.size > 0, "Input series must not be empty."
    
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()

    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate recall.
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y_hat.size > 0, "Input series must not be empty."
    
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()

    return tp / (tp + fn) if (tp + fn) > 0 else 0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (RMSE).
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y_hat.size > 0, "Input series must not be empty."
    
    sqdiff = (y - y_hat) ** 2
    mean = sqdiff.mean()
    return np.sqrt(mean)


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (MAE).
    """
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y_hat.size > 0, "Input series must not be empty."
    
    absdiff = (y_hat - y).abs()
    return absdiff.mean()
