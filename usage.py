"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(4)
# Test case 1
# Real Input and Real Output
print("-----------------Test case 1-----------------")
print("Real Input and Real Output")
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
# print(X)
# X.columns = ["A", "B", "C","D","E"]
y = pd.Series(np.random.randn(N))

# X = pd.DataFrame({
#         'A': [1, 1, 3, 0],
#         'B': [1, 2, 3, 4]
#     })
# y = pd.Series([1, 1, 0, 0])
# features = pd.Series(['A', 'B'])

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    y_hat = pd.Series(y_hat)
    tree.plot()
    tree.plotGraph()
    print(f"Criteria :, {criteria}")
    print(f"RMSE: {rmse(y_hat, y)}")
    print(f"MAE: {mae(y_hat, y)}")

print('Done')
# Test case 2
# Real Input and Discrete Output

print("-----------------Test case 2-----------------")
print("Real Input and Discrete Output")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    tree.plotGraph()
    print(f"Criteria : {criteria}")
    print(f"Accuracy: {accuracy(y_hat, y)}")
    for cls in y.unique():
        print(f"Precision:{precision(y_hat, y, cls)}")
        print(f"Recall: {recall(y_hat, y, cls)}")

print('Done')
# Test case 3
# Discrete Input and Discrete .Output

print("-----------------Test case 3-----------------")
print("Discrete Input and Discrete Output")

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    tree.plotGraph()
    print(f"Criteria :, {criteria}")
    print(f"Accuracy: , {accuracy(y_hat, y)}")
    for cls in y.unique():
        print(f"Precision:, {precision(y_hat, y, cls)}")
        print(f"Recall: , {recall(y_hat, y, cls)}")
print('Done')
# Test case 4
# Discrete Input and Real Output

print("-----------------Test case 4-----------------")
print("Discrete Input and Real Output")

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)

    tree.plot()
    tree.plotGraph()
    print(f"Criteria : {criteria}")
    print(f"RMSE: {rmse(y_hat, y)}")
    print(f"MAE: {mae(y_hat, y)}")
print('Done')