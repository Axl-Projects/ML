import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# 1. Chargement des données
print("-- Chargement des données...")
data_frame = pd.read_csv('Salary.csv')
print(f"Dataset shape: {data_frame.shape}")
print(data_frame.head())
print(f"\nStatistiques descriptives:\n{data_frame.describe()}")

# 2. Visualisation initiale
plt.figure(figsize=(12, 5))

# Subplot 1: Nuage de points
plt.subplot(1, 2, 1)
plt.scatter(data_frame['YearsExperience'], data_frame['Salary'], 
            alpha=0.6, edgecolors='k', s=80)
plt.xlabel('Années d\'expérience', fontsize=12)
plt.ylabel('Salaire ($)', fontsize=12)
plt.title('Relation Expérience - Salaire', fontsize=14)
plt.grid(True, alpha=0.3)

# Subplot 2: Distribution
plt.subplot(1, 2, 2)
sns.histplot(data_frame['Salary'], kde=True, bins=15)
plt.xlabel('Salaire ($)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.title('Distribution des salaires', fontsize=14)
plt.tight_layout()
plt.savefig('salary_distribution.png', dpi=100, bbox_inches='tight')
plt.show()

# 3. Préparation des données
X = data_frame.iloc[:, :-1].values  # Expérience
Y = data_frame.iloc[:, -1].values   # Salaire

# 4. Division train/test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, shuffle=True
)
print(f"\n-- Division train/test:")
print(f"  - Train: {len(X_train)} échantillons ({len(X_train)/len(X)*100:.0f}%)")
print(f"  - Test : {len(X_test)} échantillons ({len(X_test)/len(X)*100:.0f}%)")

# 5. Entraînement du modèle
print("\n-- Entraînement du modèle...")
model = LinearRegression()
model.fit(X_train, Y_train)

# 6. Coefficients du modèle
print(f"\n-- Équation du modèle:")
print(f"  Salaire = {model.coef_[0]:.2f} × AnnéesExpérience + {model.intercept_:.2f}")
print(f"  → Chaque année d'expérience ajoute ${model.coef_[0]:.2f} au salaire")

# 7. Prédictions et évaluation
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

# Métriques d'évaluation
def print_metrics(y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n-- Métriques sur {dataset_name}:")
    print(f"  R² Score      : {r2:.4f} ({r2*100:.1f}% de variance expliquée)")
    print(f"  MSE           : ${mse:,.0f}")
    print(f"  RMSE (erreur) : ${rmse:,.0f}")
    print(f"  MAE           : ${mae:,.0f}")
    return r2

r2_train = print_metrics(Y_train, Y_pred_train, "TRAIN")
r2_test = print_metrics(Y_test, Y_pred_test, "TEST")

# 8. Visualisation des résultats
plt.figure(figsize=(12, 5))

# Subplot 1: Droite de régression
plt.subplot(1, 2, 1)
plt.scatter(X_train, Y_train, color='blue', alpha=0.5, label='Données train', s=70)
plt.scatter(X_test, Y_test, color='green', alpha=0.7, label='Données test', s=80, edgecolors='k')
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(x_range, model.predict(x_range), color='red', linewidth=3, label='Régression')
plt.xlabel('Années d\'expérience', fontsize=12)
plt.ylabel('Salaire ($)', fontsize=12)
plt.title('Régression Linéaire - Salaire vs Expérience', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Prédictions vs Réalités
plt.subplot(1, 2, 2)
plt.scatter(Y_test, Y_pred_test, alpha=0.7, s=80, edgecolors='k')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 
         'r--', lw=2, label='Parfaite prédiction')
plt.xlabel('Salaire réel ($)', fontsize=12)
plt.ylabel('Salaire prédit ($)', fontsize=12)
plt.title(f'Prédictions vs Réalités (R²={r2_test:.3f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_results.png', dpi=100, bbox_inches='tight')
plt.show()

# 9. Prédiction d'exemple
print("\n Exemples de prédictions:")
test_experiences = [[1], [3], [5], [10], [15]]
for exp in test_experiences:
    pred = model.predict([exp])[0]
    print(f"  {exp[0]} an(s) d'expérience → ${pred:,.0f}")

# 10. Vérification d'overfitting
if abs(r2_train - r2_test) > 0.1:
    print(f"\n--  Attention : Différence significative entre R² train ({r2_train:.3f}) et test ({r2_test:.3f})")
    print("   Le modèle pourrait être en overfitting!")
else:
    print(f"\n-- Modèle stable : R² train ({r2_train:.3f}) et test ({r2_test:.3f}) similaires")

print("\n-- Analyse terminée avec succès!")