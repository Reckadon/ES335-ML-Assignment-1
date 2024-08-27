import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tree.base import *
from tree.utils import *
from time import perf_counter as time

np.random.seed(42)

def create_binary_dataset(N, M):
    # Generate a random binary matrix with N samples and M features
    data = np.random.randint(0, 2, size=(N, M))
    
    # Create column names for the features
    feature_names = [f'Feature_{i+1}' for i in range(M)]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)

    # Change datatype to float
    for i in range(M):
        df[f'Feature_{i+1}'] = df[f'Feature_{i+1}'].astype(float)

    df.rename(columns={f'Feature_{M}': 'Output'}, inplace=True)
    return df

N = 20
M = 20
iterations = 1

timeFitting = np.array([[0]*(M + 1) for i in range(N + 1)], dtype=float)
timePredicting = np.array([[0]*(M + 1) for i in range(N + 1)], dtype=float)

for n in range(1, N + 1):
    for m in range(1, M + 1):
        pred = 0
        fit = 0
        for iter in range(iterations):
            DF = create_binary_dataset(2*n, m)
            x = DF.drop(columns=['Output'])
            y = DF['Output']
            xTrain = x.head(n)
            yTrain = y.head(n)
            xTest = x.tail(n)
            yTest = y.tail(n)
            tree = DecisionTree(criterion='information_gain', maxDepth=3)
            
            start = time()
            tree.fit(xTrain, yTrain)
            end = time()
            fit += end - start
            
            start = time()
            ypred = tree.predict(xTest)
            end = time()
            pred += end - start

        timeFitting[n][m] = fit / iterations
        timePredicting[n][m] = pred / iterations

x_dim, y_dim = N + 1, M + 1
x = np.arange(x_dim)
y = np.arange(y_dim)
x, y = np.meshgrid(x, y)

# Create the plot with four subplots
fig = plt.figure(figsize=(12,8))

# Plot for timeFitting
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(x, y, timeFitting, cmap='viridis')
ax1.set_title('Time Fitting', fontsize=14)
ax1.set_xlabel('Number of Samples (N)', fontsize=12)
ax1.set_ylabel('Number of Features (M)', fontsize=12)
ax1.set_zlabel('Time (s)', fontsize=12)
ax1.tick_params(axis='both', labelsize=10)

# Plot for timePredicting
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(x, y, timePredicting, cmap='viridis')
ax2.set_title('Time Predicting', fontsize=14)
ax2.set_xlabel('Number of Samples (N)', fontsize=12)
ax2.set_ylabel('Number of Features (M)', fontsize=12)
ax2.set_zlabel('Time (s)', fontsize=12)
ax2.tick_params(axis='both', labelsize=10)

# Plot for O(n * d * log n)
O_nd_log_n = np.array([[n * m * np.log2(n + 1) for m in range(y_dim)] for n in range(x_dim)])
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot_surface(x, y, O_nd_log_n, cmap='plasma')
ax3.set_title('O(n * d * log n)', fontsize=14)
ax3.set_xlabel('Number of Samples (N)', fontsize=12)
ax3.set_ylabel('Number of Features (M)', fontsize=12)
ax3.set_zlabel('Time Complexity', fontsize=12)
ax3.tick_params(axis='both', labelsize=10)

# Plot for log2(n)
log2_n = np.array([[np.log2(n + 1) for _ in range(y_dim)] for n in range(x_dim)])
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot_surface(x, y, log2_n, cmap='coolwarm')
ax4.set_title('log2(n)', fontsize=14)
ax4.set_xlabel('Number of Samples (N)', fontsize=12)
ax4.set_ylabel('Number of Features (M)', fontsize=12)
ax4.set_zlabel('log2(n)', fontsize=12)
ax4.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.show()
