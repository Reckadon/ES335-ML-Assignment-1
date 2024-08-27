import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *

class Node:
    def __init__(self,feature,threshold,left,right,value) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self,criterion,maxDepth = 5) -> None:
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.isRealOutput = False

    def fit(self,X,y) -> None:
        X_ohe = one_hot_encoding(X)
        self.tree = self.buildTree(X_ohe,y,0)
        self.isRealOutput = isReal(y)
    
    def buildTree(self,X,y,depth) -> Node:
        node = Node(None,None,None,None,None)
        if depth >= self.maxDepth or len(y.unique()) == 1:
            node.value = y.mean() if self.isRealOutput else y.mode()[0]
            return node
        bestFeature,bestThreshold = bestSplit(X,y,self.criterion)
        if bestFeature is None:
            node.value = y.mean() if self.isRealOutput else y.mode()[0]
            return node
        leftX,rightX,leftY,rightY = split(X,y,bestThreshold,bestFeature)
        node.feature = bestFeature
        node.threshold = bestThreshold
        node.left = self.buildTree(leftX,leftY,depth+1)
        node.right = self.buildTree(rightX,rightY,depth+1)
        return node

    def predictOne(self,node,row) -> float:
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return self.predictOne(node.left,row)
        else:
            return self.predictOne(node.right,row)

    def predict(self,X) -> np.ndarray:
        X_ohe = one_hot_encoding(X)
        out = []
        for i in range(X_ohe.shape[0]):
            out.append(self.predictOne(self.tree,X_ohe.iloc[i]))

        return np.array(out)
    
    def plot(self) -> None:
        def plot_node(node, depth):
            indent = "    " * depth  # Indentation for each level of depth
            if node.value is not None:
                print(f"{indent}Class {node.value}")
                return
            
            # Print the decision condition at the current node
            print(f"{indent}?({node.feature} > {node.threshold})")
            
            # Print the left (Y: Yes) branch
            print(f"{indent}Y: ", end="")
            plot_node(node.left, depth + 1)
            
            # Print the right (N: No) branch
            print(f"{indent}N: ", end="")
            plot_node(node.right, depth + 1)
        
        # Start plotting from the root of the tree
        plot_node(self.tree, 0)

    def plotGraph(self) -> None:
        # Helper function to recursively plot the tree
        def plot_node(node, depth, x, y, dx):
            if node is None:
                return
            
            # Plot the current node
            plt.text(x, y, f'{node.feature}\n<= {node.threshold}' if node.value is None else f'Value: {node.value:.2f}',
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
            # Calculate the positions of the child nodes
            if node.left is not None:
                plt.plot([x, x - dx], [y, y - 1], 'k-')
                plot_node(node.left, depth + 1, x - dx, y - 1, dx / 2)
                
            if node.right is not None:
                plt.plot([x, x + dx], [y, y - 1], 'k-')
                plot_node(node.right, depth + 1, x + dx, y - 1, dx / 2)

        # Initial call to the helper function
        plt.figure(figsize=(12, 8))
        plot_node(self.tree, 0, 0, 0, 1)
        plt.axis('off')
        plt.show()