import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def load_data(url):
    """Load and clean the dataset"""
    df = pd.read_excel(url, engine="xlrd")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"\(", "", regex=True)
        .str.replace(r"\)", "", regex=True)
        .str.replace(r",", "", regex=True)
    )

    df = df.rename(columns={
        'cement_component_1kg_in_a_m3_mixture': 'cement',
        'water_component_4kg_in_a_m3_mixture': 'water',
        'age_day': 'age',
        'concrete_compressive_strengthmpa_megapascals': 'strength'
    })

    return df


def train_polynomial_regression(X, y, degree):
    """Train polynomial regression model"""
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)

    return model, poly, r2


def main():
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

    # Load data
    df = load_data(url)

    # Features and target
    X = df[['cement', 'water', 'age']].values
    y = df['strength'].values

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Polynomial degree
    degree = 3

    # Train model
    model, poly, r2_train = train_polynomial_regression(X_train, y_train, degree)

    # Test performance
    X_test_poly = poly.transform(X_test)
    y_test_pred = model.predict(X_test_poly)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"Polynomial Regression (degree={degree})")
    print(f"R² Train: {r2_train:.2f}")
    print(f"R² Test : {r2_test:.2f}")

    # Visualization (age vs strength)
    age = X_test[:, 2]
    sorted_idx = age.argsort()

    plt.scatter(age, y_test, label="Actual values", alpha=0.6)
    plt.plot(age[sorted_idx], y_test_pred[sorted_idx], color="red", label="Prediction")
    plt.xlabel("Age (days)")
    plt.ylabel("Concrete compressive strength (MPa)")
    plt.title("Polynomial Regression on Concrete Strength")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
