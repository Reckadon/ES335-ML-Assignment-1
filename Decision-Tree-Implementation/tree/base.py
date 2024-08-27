import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *

np.random.seed(42)

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.predictedValue = None
        self.children = {}

class DecisionTree:
    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        category = "category"
        featureDataTypes = []
        for col in X.columns:
            featureDataTypes.append(X[col].dtype.name)

        # for discrete input and output
        if category in featureDataTypes and y.dtype.name == category:
            IOtype = ("discrete", "discrete")
        # for discrete input and real output
        elif category in featureDataTypes and y.dtype.name != category:
            IOtype = ("discrete", "real")
        # for real input and discrete output
        elif category not in featureDataTypes and y.dtype.name == category:
            IOtype = ("real", "discrete")
        # for real input and output
        else:
            IOtype = ("real", "real")

        features = X.columns
        depth = 0
        self.tree = self.buildTree(X, y, features, depth, IOtype)

    def buildTree(self, X, y, features, depth, IOtype):
        # Create a new node
        node = Node()

        # If the stopping criteria is met, return a leaf node
        if depth == self.max_depth or len(y.unique()) == 1 or len(X) == 0:
            node.predictedValue = y.mean() if IOtype[1] == "real" else y.mode()[0]
            return node

        # Find the best feature and value to split on
        best_feature, best_val = bestSplit(X, y, self.criterion,IOtype)

        # If no split improves the model, return a leaf node
        if best_feature is None:
            node.predictedValue = y.mean() if IOtype[1] == "real" else y.mode()[0]
            return node

        # Set the feature and threshold for this node
        node.feature = best_feature
        node.threshold = best_val

        # Split the data
        X_left, X_right, y_left, y_right = splitData(X, y, best_feature, best_val,IOtype)

        # Recursively build the left and right subtrees
        node.left = self.buildTree(X_left, y_left, features, depth + 1, IOtype)
        node.right = self.buildTree(X_right, y_right, features, depth + 1, IOtype)

        return node

    def predictSingle(self, node, x):
        # Recursively traverse the tree to make a prediction
        if node.predictedValue is not None:
            return node.predictedValue

        if x[node.feature] <= node.threshold:
            return self.predictSingle(node.left, x)
        else:
            return self.predictSingle(node.right, x)

    def predict(self, X):
        # Predict the target for each instance in X
        return X.apply(lambda x: self.predictSingle(self.tree, x), axis=1)

    def score(self, X, y):
        # Calculate accuracy for classification or MSE for regression
        predictions = self.predict(X)
        if self.criterion in ['entropy', 'gini_index']:
            return np.mean(predictions == y)
        elif self.criterion == 'mse':
            return np.mean((predictions - y) ** 2)

    def plot(self):
        # Optional: Method to plot the decision tree (using matplotlib, etc.)
        pass
