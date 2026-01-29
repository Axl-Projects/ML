import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


def load_data(path):
    """Load diabetes dataset"""
    return pd.read_csv(path)


def visualize_data(df):
    """Simple visualization: Age vs Outcome"""
    plt.scatter(df['Age'], df['Outcome'], alpha=0.6)
    plt.xlabel("Age")
    plt.ylabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
    plt.title("Age vs Diabetes Outcome")
    plt.show()


def prepare_data(df):
    """Split features and target"""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y


def train_model(X_train, y_train):
    """Train Logistic Regression model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        solver='liblinear',
        random_state=0,
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model performance"""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", model.score(X_test_scaled, y_test))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_patient(model, scaler, patient_data):
    """
    Predict diabetes risk for a single patient
    patient_data: list of 8 values
    """
    patient_df = pd.DataFrame([patient_data], columns=[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ])

    patient_scaled = scaler.transform(patient_df)

    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]

    return prediction, probability


def main():
    # Load dataset
    df = load_data("diabetes.csv")

    # Display first rows
    print(df.head())

    # Visualization
    visualize_data(df)

    # Prepare data
    X, y = prepare_data(df)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train model
    model, scaler = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, scaler, X_test, y_test)

    # ===== TEST CASES =====

    print("\n--- Patient Tests ---")

    # Patient 1
    patient_1 = [1, 90, 70, 20, 80, 22.0, 0.2, 25]
    pred, prob = predict_patient(model, scaler, patient_1)
    print(f"Patient 1 → Diabetes: {pred} | Probability: {prob:.2f}")

    # Patient 2
    patient_2 = [5, 180, 85, 35, 200, 33.5, 0.8, 50]
    pred, prob = predict_patient(model, scaler, patient_2)
    print(f"Patient 2 → Diabetes: {pred} | Probability: {prob:.2f}")

    # Patient 3
    patient_3 = [2, 130, 75, 25, 120, 28.0, 0.5, 35]
    pred, prob = predict_patient(model, scaler, patient_3)
    print(f"Patient 3 → Diabetes: {pred} | Probability: {prob:.2f}")


if __name__ == "__main__":
    main()
