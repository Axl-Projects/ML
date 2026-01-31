# Titanic Survival Prediction using Decision Tree

## -- Project Overview
This project uses a **Decision Tree Classifier** to predict whether a passenger survived the Titanic disaster based on personal and ticket information.

The goal is to understand how machine learning models make decisions using structured data.

---

## -- Dataset
The dataset comes from the famous **Titanic Kaggle competition**.

### Features used:
- `Pclass` : Passenger class (1st, 2nd, 3rd)
- `Sex` : Gender (male / female)
- `Age` : Passenger age
- `Fare` : Ticket price

### Target:
- `Survived` (0 = No, 1 = Yes)

---

## -- Data Preprocessing
- Selected relevant columns
- Filled missing age values using the median
- Encoded categorical variable `Sex`
- Split data into training and testing sets

---

## -- Model
- Algorithm: **Decision Tree Classifier**
- Library: `scikit-learn`
- Train/Test split: 80% / 20%
- Random state fixed for reproducibility

---

## -- Run Analysis

1. Install dependencies
pip install pandas numpy scikit-learn

2. Run the script
python decision_tree_titanic.py 

---

## -- Model Evaluation
The model accuracy is evaluated on the test set using:

```python
model.score(X_test, y_test)

