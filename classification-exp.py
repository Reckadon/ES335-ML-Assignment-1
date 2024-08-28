import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

# from HAR.MakeDataset import X_train, y_train
from tree.utils import *
from tree.base import *
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting

plt.scatter(X[:, 0], X[:, 1], c=y)
xtrain, xtest, ytrain, ytest = train_test_split(X[:, 0], y, test_size=0.3)
xtrain = pd.DataFrame(xtrain)
ytrain = pd.Series(ytrain)
ytest = pd.Series(ytest)
xtest = pd.DataFrame(xtest)
tree = DecisionTree(criterion="information_gain")
tree.fit(xtrain, ytrain)
y_hat = tree.predict(xtest)
print("Accuracy of our Decision tree on first class is " + str(accuracy(ytest, y_hat) * 100) + "%")
for cls in ytest.unique():
    print(f"Precision:{precision(y_hat, ytest, cls)}")
    print(f"Recall: {recall(y_hat, ytest, cls)}")

xtrain2, xtest2, ytrain2, ytest2 = train_test_split(X[:, 1], y, test_size=0.3)
xtrain2 = pd.DataFrame(xtrain2)
ytrain2 = pd.Series(ytrain2)
ytest2 = pd.Series(ytest2)
xtest2 = pd.DataFrame(xtest2)
tree = DecisionTree(criterion="information_gain")
tree.fit(xtrain2, ytrain2)
y_hat = tree.predict(xtest2)
print("Accuracy of our Decision tree on second class is " + str(accuracy(y_hat, ytest2) * 100) + "%")
for cls in ytest2.unique():
    print(f"Precision:{precision(y_hat, ytest2, cls)}")
    print(f"Recall: {recall(y_hat, ytest2, cls)}")

n_folds = 5
dataset_size = len(X)
indices = np.arange(dataset_size)
np.random.shuffle(indices)

fold_sizes = np.full(n_folds, dataset_size // n_folds)
fold_sizes[:dataset_size % n_folds] += 1

current = 0
folds = []
for fold_size in fold_sizes:
    start, stop = current, current + fold_size
    folds.append((indices[start:stop]))
    current = stop

fold_accuracies = []
print("For first class:")
for i in range(n_folds):
    # Create the training and validation sets
    train_indices = np.concatenate([folds[j] for j in range(n_folds) if j != i])
    val_indices = folds[i]

    X_train, y_train = pd.DataFrame(X[train_indices, 0]), pd.Series(y[train_indices])
    X_val, y_val = pd.DataFrame(X[val_indices, 0]), pd.Series(y[val_indices])
    tree.fit(X_train, y_train)
    acc = sum(y_val == tree.predict(X_val)) / len(y_val)
    fold_accuracies.append(acc)
    print(f"Fold {i + 1}, Accuracy: {acc:.4f}")

print(f"Mean accuracy is {np.mean(fold_accuracies)}")

print("For second class:")
fold_accuracies = []
for i in range(n_folds):
    # Create the training and validation sets
    train_indices = np.concatenate([folds[j] for j in range(n_folds) if j != i])
    val_indices = folds[i]

    X_train, y_train = pd.DataFrame(X[train_indices, 1]), pd.Series(y[train_indices])
    X_val, y_val = pd.DataFrame(X[val_indices, 1]), pd.Series(y[val_indices])
    tree.fit(X_train, y_train)
    acc = sum(y_val == tree.predict(X_val)) / len(y_val)
    fold_accuracies.append(acc)
    print(f"Fold {i + 1}, Accuracy: {acc:.4f}")

print(f"Mean accuracy is {np.mean(fold_accuracies)}")

def splitfold(x, n):
    i = np.arange(len(x))
    np.random.shuffle(i)
    f = np.array_split(i, n)
    return f


def nestedvalid(x, y,cl, depths=[2, 3, 4, 5, 6], ofold=5, ifold=5):
    oscore = []
    ofold_indices = splitfold(x, ofold)
    for outer_train_indices in ofold_indices:
        outer_test_indices = np.concatenate(
            [outer_train_indices for idx in range(ofold) if not np.array_equal(outer_train_indices, idx)])
        X_train, X_test = X[outer_train_indices, cl], X[outer_test_indices, cl]
        y_train, y_test = y[outer_train_indices], y[outer_test_indices]
        inner_folds_indices = splitfold(X_train, ifold)
        best_depth = None
        best_score = -np.inf
        for depth in depths:
            iscore = []
            for inner_train_indices in inner_folds_indices:
                inner_val_indices = np.concatenate(
                    [inner_train_indices for idx in range(ifold) if not np.array_equal(inner_train_indices, idx)])
                X_inner_train, X_inner_val = X_train[inner_train_indices], X_train[inner_val_indices]
                y_inner_train, y_inner_val = y_train[inner_train_indices], y_train[inner_val_indices]
                tree = DecisionTree(criterion="information_gain", maxDepth=depth)
                tree.fit(X_inner_train, y_inner_train)
                acc = sum(y_inner_val == tree.predict(X_inner_val)) / len(y_inner_val)
                iscore.append(acc)
            mean_score = np.mean(iscore)
            if mean_score > best_score:
                best_score = mean_score
                best_depth = depth
        final_model = DecisionTree(criterion='information_gain', maxDepth=best_depth)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        oscore.append(accuracy(y_test, y_pred))

    return np.mean(oscore) * 100, np.std(oscore)


mscore, stdcore = nestedvalid(X, y,0, depths=[2, 3, 4, 5, 6], ofold=5, ifold=5)
print(str(mscore) + " % Accuracy for first class")
print("Standard Deviation is " + str(stdcore))

mscore, stdcore = nestedvalid(X, y,1, depths=[2, 3, 4, 5, 6], ofold=5, ifold=5)
print(str(mscore) + " % Accuracy for second class")
print("Standard Deviation is " + str(stdcore))