# -- Diabetes Prediction using Logistic Regression

This project uses **Logistic Regression** to predict whether a patient is likely to have diabetes based on medical diagnostic measurements.

The dataset used is the **Pima Indians Diabetes Dataset**, a well-known dataset for binary classification problems in machine learning.

---

## -- Dataset
**Name:** Pima Indians Diabetes Dataset  
**Source:** Kaggle / UCI Machine Learning Repository  

### Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target:
- `Outcome`
  - `0` → Non-diabetic
  - `1` → Diabetic

---

## -- Model Used
- **Logistic Regression**
- Solver: `liblinear`
- Suitable for binary classification problems
- Sensitive to feature scaling (Standardization applied)

---

## -- Workflow
1. Load and explore the dataset
2. Visualize basic relationships (Age vs Outcome)
3. Split data into training and test sets
4. Apply feature scaling using `StandardScaler`
5. Train a Logistic Regression model
6. Evaluate model performance
7. Test cases

---

## -- Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

These metrics help better understand model performance, especially for medical classification tasks.

---

## -- How to Run

### Run analysis

python LogisticRegression.py

### 1. Install dependencies
```bash
pip install pandas matplotlib scikit-learn
