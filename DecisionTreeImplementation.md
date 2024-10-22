## Graphs and Results from `usage.py`

### Real Input and Real Output

- **With Information Gain as Criteria**  
  ![Information Gain - Real Input, Real Output](Images/image.png)
  - **RMSE**: 0.6365
  - **MAE**: 0.3944

- **With Gini Index as Criteria**  
  ![Gini Index - Real Input, Real Output](Images/image-1.png)
  - **RMSE**: 0.8659
  - **MAE**: 0.4314

### Real Input and Discrete Output

- **With Information Gain as Criteria**  
  ![Information Gain - Real Input, Discrete Output](Images/image-6.png)
  - **Accuracy**: 0.9333
  - **Precision**: 0.8
  - **Recall**: 0.8

- **With Gini Index as Criteria**  
  ![Gini Index - Real Input, Discrete Output](Images/image-7.png)
  - **Accuracy**: 0.8333
  - **Precision**: 0.6667
  - **Recall**: 0.4

### Discrete Input and Discrete Output

- **With Information Gain as Criteria**  
  ![Information Gain - Discrete Input, Discrete Output](Images/image-2.png)
  - **Accuracy**: 0.8
  - **Precision**: 0.7778
  - **Recall**: 0.875

- **With Gini Index as Criteria**  
  ![Gini Index - Discrete Input, Discrete Output](Images/image-3.png)
  - **Accuracy**: 0.7
  - **Precision**: 0.5385
  - **Recall**: 0.875

### Discrete Input and Real Output

- **With Information Gain as Criteria**  
  ![Information Gain - Discrete Input, Real Output](Images/image-4.png)
  - **RMSE**: 0.7630
  - **MAE**: 0.3860

- **With Gini Index as Criteria**  
  ![Gini Index - Discrete Input, Real Output](Images/image-5.png)
  - **RMSE**: 1.2383
  - **MAE**: 0.6397

---

## Decision Tree Classifier Performance from `classification-exp.py`

### First Class

**Overall Metrics**
- **Accuracy**: 90.0%
- **Precision**: 0.9286
- **Recall**: 0.8667
- **Second Precision**: 0.8750
- **Second Recall**: 0.9333

**Cross-Validation Accuracy**
- **Fold 1**: 0.8500
- **Fold 2**: 0.5000
- **Fold 3**: 0.8000
- **Fold 4**: 1.0000
- **Fold 5**: 0.8000
- **Mean Accuracy**: 0.79

**Additional Statistics**
- **Overall Accuracy**: 93.0%
- **Standard Deviation**: 0.0400

### Second Class

**Overall Metrics**
- **Accuracy**: 50.0%
- **Precision**: 0.5789
- **Recall**: 0.6111
- **Second Precision**: 0.3636
- **Second Recall**: 0.3333

**Cross-Validation Accuracy**
- **Fold 1**: 0.7000
- **Fold 2**: 0.5500
- **Fold 3**: 0.5000
- **Fold 4**: 0.8000
- **Fold 5**: 0.5000
- **Mean Accuracy**: 0.61

**Additional Statistics**
- **Overall Accuracy**: 73.0%
- **Standard Deviation**: 0.0980

---

## Results from `auto-efficiency.py`

**Our Model**

- **RMSE**: 4.189785650539697
  
**SK-Learn Model**

- **RMSE**: 3.3806036461937135

---

## Runtime Complexity Analysis from `experiments.py`

### For Fitting 

The time complexity for fitting a dataset into a Decision Tree is **O(n * m * log(n))** where **n** is the number of rows in the dataset and **m** is the number of features used for fitting.

For predicting, the time complexity is of the order **O(log(n))** or **O(h)**, where **h** is the maximum depth of the decision tree, whichever is lower.

Here is a plot from our analysis:
![Runtime Complexity Analysis](Images/plotTree-Q4.png) 

The plot for both fitting and predicting is matching the shape of theoretical graphs in essence.

---
