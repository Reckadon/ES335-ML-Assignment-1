import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tree.base import *
from time import time


np.random.seed(42)

def create_binary_dataset(N, M):
    # Generate a random binary matrix with N samples and M features
    data = np.random.randint(0, 2, size=(N, M))
    
    # Create column names for the features
    feature_names = [f'Feature_{i+1}' for i in range(M)]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df.rename(columns={f'Feature_{M}': 'Output'}, inplace=True)
    return df

timeFitting = [[0]*100 for i in range(100)]
timePredicting = [[0]*100 for i in range(100)]

for n in range(1,100):
    for m in range(1,100):
        trainDF = create_binary_dataset(n, m)
        X_train = trainDF.drop(columns='Output')
        y_train = trainDF['Output']
        tree = DecisionTree(criterion = 'information_gain', maxDepth = 5)
        start = time()
        tree.fit(X_train, y_train)
        end = time()
        timeFitting[n][m] = end - start
        testDF = create_binary_dataset(n+1, m+1)
        X_test = testDF.drop(columns='Output')
        y_test = testDF['Output']
        start = time()
        tree.predict(X_test)
        end = time()
        timePredicting[n][m] = end - start

x_dim, y_dim = timeFitting.shape

# Create the X and Y indices
x = np.arange(x_dim)
y = np.arange(y_dim)
x, y = np.meshgrid(x, y)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, timeFitting, cmap='viridis')

# Label the axes
ax.set_xlabel('X Index')
ax.set_ylabel('Y Index')
ax.set_zlabel('Z Value')

# Show the plot
plt.show()
