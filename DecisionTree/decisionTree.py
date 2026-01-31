import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Charger le dataset
df = pd.read_csv('train.csv')

# Selection des colonnes utiles
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Gestion des valeurs manquantes
df['Age'].fillna(df['Age'].median())

# Encodage des variables catégorielles
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Séparer X et y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Creation jeu d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construire un arbr e de decision
modelTree = tree.DecisionTreeClassifier()
modelTree.fit(X_train, y_train)


# ============================
# TEST CASES - PRÉDICTIONS
# ============================

# Cas 1 : Femme, 1ère classe, 29 ans, billet cher
test_case_1 = pd.DataFrame({
    'Pclass': [1],
    'Sex': [1],   # female
    'Age': [29],
    'Fare': [100]
})

# Cas 2 : Homme, 3ème classe, 40 ans, billet pas cher
test_case_2 = pd.DataFrame({
    'Pclass': [3],
    'Sex': [0],   # male
    'Age': [40],
    'Fare': [7.25]
})

# Cas 3 : Enfant, 2ème classe, femme
test_case_3 = pd.DataFrame({
    'Pclass': [2],
    'Sex': [1],
    'Age': [8],
    'Fare': [20]
})

# Liste des tests
test_cases = [test_case_1, test_case_2, test_case_3]

# Prédictions
for i, case in enumerate(test_cases, start=1):
    prediction = modelTree.predict(case)[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    print(f"Test case {i}: {result}")

# Calculer le taux de performance du modèle
print ('Accuracy Test: ', modelTree.score(X_test, y_test))
