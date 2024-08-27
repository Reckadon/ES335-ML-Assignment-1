import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(44)


# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])



print(data)

data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

data['horsepower'] = data['horsepower'].astype(float)


data = data.drop(columns=["car name"])

data['origin'] = data['origin'].astype('category').cat.codes

X = data.drop(columns=["mpg"])
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(criterion='gini_index',maxDepth=6)

tree.fit(X_train, y_train)
ypred = tree.predict(X_test)
print(rmse(y_test, ypred))

clf = DecisionTreeRegressor(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(rmse(y_test, y_pred))

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

