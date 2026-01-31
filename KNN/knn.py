import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Charger le dataset
df_bc = load_breast_cancer()
X, y = df_bc.data, df_bc.target

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser le dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Créer le modèle knn
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# ============================
# TEST CASES - PREDICTIONS
# ============================

# Cas 1 : tumeur probablement bénigne
test_case_1 = X_test[0].reshape(1, -1)

# Cas 2 : tumeur probablement maligne
test_case_2 = X_test[1].reshape(1, -1)

test_cases = [test_case_1, test_case_2]

for i, case in enumerate(test_cases, start=1):
    prediction = knn.predict(case)[0]
    result = "Malignant" if prediction == 0 else "Benign"
    print(f"Test case {i}: {result}")


# Taux de précision
score = knn.score(X_test, y_test)
print(f"Accuracy: {score:.4f} ~ {score * 100:.2f}%")
