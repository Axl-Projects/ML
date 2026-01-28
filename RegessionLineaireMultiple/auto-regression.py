import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.25

def load_and_prepare_data(filepath):
    """Charge et prépare le dataset Auto MPG"""
    print("-- Chargement des données...")
    
    # Définition des colonnes
    columns = [
        "mpg", "cylinders", "displacement", "horsepower", 
        "weight", "acceleration", "model_year", "origin", "car_name"
    ]
    
    # Chargement avec gestion des valeurs manquantes
    df = pd.read_csv(filepath, sep="\\s+", names=columns, na_values="?")
    
    print(f"Dataset shape avant cleaning: {df.shape}")
    
    # Nettoyage
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    
    df = df.dropna()
    
    print(f"Dataset shape après cleaning: {df.shape}")
    print(f"Valeurs manquantes: \n{df.isnull().sum()}")
    
    df['origin'] = df['origin'].astype('category')
    
    return df

def explore_data(df):
    """Analyse exploratoire des données"""
    print("\n-- Analyse exploratoire:")
    print("=" * 50)
    
    # Statistiques descriptives
    print("\n-- Statistiques descriptives:")
    print(df.describe())
    
    # Distribution de la target
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['mpg'], kde=True, bins=30)
    plt.title('Distribution de MPG (target)')
    plt.xlabel('MPG')
    plt.ylabel('Fréquence')
    
    # Corrélations
    plt.subplot(1, 2, 2)
    numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                    'weight', 'acceleration', 'model_year']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True)
    plt.title('Matrice de Corrélation')
    plt.tight_layout()
    plt.savefig('mpg_correlation.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Corrélations avec MPG
    print("\n-- Corrélations avec MPG:")
    mpg_corr = corr_matrix['mpg'].abs().sort_values(ascending=False)
    print(mpg_corr[1:6])
    
    # Visualisation des relations clés
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].scatter(df['weight'], df['mpg'], alpha=0.5)
    axes[0, 0].set_xlabel('Poids (lbs)')
    axes[0, 0].set_ylabel('MPG')
    axes[0, 0].set_title('MPG vs Poids')
    
    axes[0, 1].scatter(df['horsepower'], df['mpg'], alpha=0.5)
    axes[0, 1].set_xlabel('Puissance (hp)')
    axes[0, 1].set_ylabel('MPG')
    axes[0, 1].set_title('MPG vs Puissance')
    
    axes[1, 0].scatter(df['displacement'], df['mpg'], alpha=0.5)
    axes[1, 0].set_xlabel('Cylindrée (cu. in.)')
    axes[1, 0].set_ylabel('MPG')
    axes[1, 0].set_title('MPG vs Cylindrée')
    
    axes[1, 1].scatter(df['model_year'], df['mpg'], alpha=0.5)
    axes[1, 1].set_xlabel('Année modèle')
    axes[1, 1].set_ylabel('MPG')
    axes[1, 1].set_title('MPG vs Année')
    
    plt.tight_layout()
    plt.savefig('mpg_relationships.png', dpi=100, bbox_inches='tight')
    plt.show()

def build_model_pipeline():
    """Construit le pipeline de preprocessing et modèle"""
    
    # Colonnes catégorielles et numériques
    categorical_cols = ['origin', 'car_name']
    numerical_cols = ['cylinders', 'displacement', 'horsepower', 
                      'weight', 'acceleration', 'model_year']
    
    # Préprocessing
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])
    
    # Pipeline complet
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test, feature_names=None):
    """Évalue le modèle et affiche les résultats"""
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n-- Résultats d'évaluation:")
    print("=" * 50)
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} MPG")
    print(f"MAE: {mae:.2f} MPG")
    
    # Visualisation prédictions vs réalité
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Parfait')
    plt.xlabel('MPG Réel')
    plt.ylabel('MPG Prédit')
    plt.title(f'Prédictions vs Réalité (R²={r2:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Résidus
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('MPG Prédit')
    plt.ylabel('Résidus (Réel - Prédit)')
    plt.title('Analyse des Résidus')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mpg_predictions_evaluation.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return r2, rmse, mae

def interpret_model(model, feature_names):
    """Interprète les coefficients du modèle"""
    print("\n-- Interprétation du modèle:")
    print("=" * 50)
    
    try:
        # Récupérer les coefficients
        coefficients = model.named_steps['regressor'].coef_
        
        # Récupérer les noms de features après preprocessing
        cat_encoder = model.named_steps['preprocessor'].transformers_[0][1]
        num_features = ['cylinders', 'displacement', 'horsepower', 
                        'weight', 'acceleration', 'model_year']
        
        # Noms des features catégorielles encodées
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features = cat_encoder.get_feature_names_out(['origin', 'car_name'])
            all_features = list(cat_features) + num_features
        else:
            all_features = feature_names
        
        # Créer DataFrame des coefficients
        coeff_df = pd.DataFrame({
            'Feature': all_features,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nTop 10 features les plus importantes:")
        print(coeff_df.head(10))
        
        # Visualisation
        plt.figure(figsize=(12, 6))
        top_20 = coeff_df.head(20)
        colors = ['green' if c > 0 else 'red' for c in top_20['Coefficient']]
        plt.barh(range(len(top_20)), top_20['Abs_Coefficient'], color=colors)
        plt.yticks(range(len(top_20)), top_20['Feature'])
        plt.xlabel('Importance (|Coefficient|)')
        plt.title('Top 20 Features les plus importantes')
        plt.tight_layout()
        plt.savefig('mpg_feature_importance.png', dpi=100, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"--  Impossible d'interpréter les coefficients: {e}")

def save_model(model, filename='mpg_predictor.pkl'):
    """Sauvegarde le modèle entraîné"""
    joblib.dump(model, filename)
    print(f"\n Modèle sauvegardé: {filename}")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("-- PRÉDICTION MPG - RÉGRESSION MULTIPLE")
    print("=" * 60)
    
    # 1. Charger les données
    df = load_and_prepare_data('dataset/auto-mpg.data-original')
    
    # 2. Analyse exploratoire
    explore_data(df)
    
    # 3. Préparer X et y
    X = df.drop('mpg', axis=1)
    y = df['mpg']
    
    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"\n Division train/test:")
    print(f"  Train: {X_train.shape[0]} échantillons")
    print(f"  Test : {X_test.shape[0]} échantillons")
    
    # 5. Construire et entraîner le modèle
    print("\n-- Construction du pipeline...")
    model_pipeline = build_model_pipeline()
    
    print("Entraînement du modèle...")
    model_pipeline.fit(X_train, y_train)
    
    # 6. Évaluation
    r2, rmse, mae = evaluate_model(model_pipeline, X_test, y_test, X.columns.tolist())
    
    # 7. Interprétation
    interpret_model(model_pipeline, X.columns.tolist())
    
    # 8. Sauvegarde
    save_model(model_pipeline)
    
    # 9. Exemples de prédictions
    print("\n-- Exemples de prédiction:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        actual = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        features = X_test.iloc[idx:idx+1] if hasattr(X_test, 'iloc') else X_test[idx:idx+1]
        predicted = model_pipeline.predict(features)[0]
        
        print(f"  Véhicule: {features['car_name'].values[0] if 'car_name' in features else 'Unknown'}")
        print(f"    - MPG réel: {actual:.1f}")
        print(f"    - MPG prédit: {predicted:.1f}")
        print(f"    - Erreur: {abs(actual-predicted):.1f} MPG")
        print()
    
    print("=" * 60)
    print("-- ANALYSE TERMINÉE AVEC SUCCÈS!")
    print("=" * 60)

if __name__ == "__main__":
    main()