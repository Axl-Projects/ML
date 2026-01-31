# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## -- Project Overview
This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify breast cancer tumors as **benign** or **malignant** based on medical features.

The objective is to understand distance-based classification and evaluate the impact of the number of neighbors on model performance.

---

## -- Dataset
The dataset is provided by **scikit-learn** and is based on real diagnostic measurements.

- Number of samples: 569
- Number of features: 30
- Classes:
  - 0: Malignant
  - 1: Benign

---

## -- Data Preparation
- Dataset loaded from `sklearn.datasets`
- Features and target separated
- Data split into training and testing sets (80% / 20%)
- Fixed random state for reproducibility

---

## -- Model
- Algorithm: K-Nearest Neighbors (KNN)
- Library: scikit-learn
- Distance metric: Euclidean (default)
- Number of neighbors: 7

---

## -- How to Run the project

1. Install dependencies
pip install numpy pandas scikit-learn matplotlib

2. Run the script
python knn.py

## -- Model Evaluation
Model performance is evaluated using accuracy:

```python
knn.score(X_test, y_test)

