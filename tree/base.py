"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int = 4 # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.t_ = None   # Stores the decision tree. Not to be accessed outside class.`

    def fit(self, X: pd.DataFrame, y: pd.Series,depth=0) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.

        # Base cases : Max depth reached, No further features to split on, All points have same y value.
        if not check_ifreal(y):
            if depth >= self.max_depth:
                self.t_ = y.mode([0])
                return
            if X.empty or X.shape[1] == 0:
                self.t_ = y.mode([0])
                return
            if y.nunique() == 1:
                self.t_ = y.loc[0]
                return
        else:
            if depth >= self.max_depth:
                self.t_ = np.mean(y)
                return
            if X.empty or X.shape[1] == 0:
                self.t_ = np.mean(y)
                return
            if y.nunique() == 1:
                self.t_ = y.loc[0]
                return



        attribute,value = opt_split_attribute(X,y,self.criterion,X.columns)
        self.t_ = {attribute: {}}
        uval = X[attribute].unique()





        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
