# -- Concrete Compressive Strength Prediction

This project applies **Polynomial Regression** to predict the **compressive strength of concrete** based on its components and age.

The dataset comes from the **UCI Machine Learning Repository** and contains quantitative measurements of concrete mixtures.

---

## -- Dataset
**Source:** UCI Machine Learning Repository  
**Features used:**
- Cement (kg/m³)
- Water (kg/m³)
- Age (days)

**Target variable:**
- Concrete compressive strength (MPa)

---

## -- Model
- Polynomial Regression (scikit-learn)
- Tested with different polynomial degrees
- Best performance observed with **degree = 3**

**Evaluation metric:**
- R² Score

---

## -- Results
- Polynomial Degree: **3**
- R² ≈ **70%**
- Shows a non-linear relationship between concrete components, age, and strength

---

## Run Analysis
python polynomiale_regression.py

## -- How to Run
1. Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn xlrd
