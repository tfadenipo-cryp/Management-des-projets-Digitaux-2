"""
Evaluate Claim Probability Model (Logistic Regression)

This script:
1.  Loads data and creates a NEW target variable: 'had_claim' (1/0).
2.  Applies preprocessing and splits into Train (80%) / Test (20%).
3.  Filters for the 14 "premium" features (un bon point de départ).
4.  Trains a Logistic Regression model (GLM Binomial) on the Train set.
5.  Makes predictions on the Test set.
6.  Calculates and prints performance (Accuracy, Precision, Recall, AUC).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# --- Add Project Root to sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# --- End of sys.path modification ---

try:
    from src.functions.engineering import engineering
    from src.models.preprocessing_premium import (
        preprocess_data_for_modeling,
    )
except ImportError as e:
    print(
        f"Error: Could not import necessary modules. {e}"
    )
    sys.exit(1)

# Les 14 features du modèle PREMIUM (bon point de départ)
FINAL_FEATURES = [
    'n_doors_0', 'vehicle_age', 'value_vehicle', 'payment_0', 'type_risk_4',
    'second_driver_0', 'driving_experience', 'type_fuel_0',
    'policies_in_force', 'lapse', 'area_0', 'seniority',
    'distribution_channel_0', 'type_risk_1'
]

def evaluate_probability_model() -> None:
    """
    Trains and evaluates the GLM Binomial (Logistic) model.
    """
    print("--- Starting PROBABILITY Model Evaluation ---")

    # 1. Load data
    print("Step 1: Loading base data...")
    base_df = engineering()
    if base_df is None or base_df.empty:
        print("❌ Failed to load base data.")
        return

    # 2. CRÉER LA NOUVELLE CIBLE (y)
    # Notre cible n'est plus 'cost_claims_year', mais 'had_claim' (1 ou 0)
    print("Step 2: Creating new target variable 'had_claim'...")
    base_df['had_claim'] = (base_df['n_claims_year'] > 0).astype(int)
    
    # S'assurer que les features de risque ne sont pas utilisées pour tricher
    base_df = base_df.drop(columns=['n_claims_history', 'r_claims_history'], errors='ignore')


    # 3. Preprocess and split
    print("Step 3: Preprocessing and splitting data (80% Train / 20% Test)...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        all_feature_names,
    ) = preprocess_data_for_modeling(
        base_df, target_column="had_claim", test_size=0.2, random_state=42
    )

    if X_train is None or X_train.empty:
        print("❌ Preprocessing failed.")
        return

    # 4. Assign feature names
    X_train.columns = all_feature_names
    X_test.columns = all_feature_names

    print(f"Step 4: Filtering for {len(FINAL_FEATURES)} selected features...")
    for f in FINAL_FEATURES:
        if f not in X_train.columns:
            X_train[f] = 0
            X_test[f] = 0

    X_train_final = X_train[FINAL_FEATURES]
    X_test_final = X_test[FINAL_FEATURES]
    
    X_train_with_const = sm.add_constant(X_train_final, prepend=True)
    X_test_with_const = sm.add_constant(X_test_final, prepend=True)

    # 5. Train Logistic Regression model
    print(f"Step 5: Training GLM (Binomial) model on {len(X_train_with_const)} rows...")
    # Binomial (ou "Logit") est le standard pour prédire OUI/NON
    model = sm.GLM(
        y_train,
        X_train_with_const,
        family=sm.families.Binomial(link=sm.families.links.logit()),
    )
    results = model.fit()
    print("✅ Model training complete.")
    print(results.summary())

    # 6. Make predictions ON TEST DATA
    print(f"\nStep 6: Making predictions on {len(X_test_with_const)} unseen test rows...")
    # Le modèle prédit la probabilité (entre 0.0 et 1.0)
    y_probabilities = results.predict(X_test_with_const)
    
    
    # --- DÉBUT DE LA CORRECTION : CALIBRAGE DU SEUIL ---
    # Un seuil de 0.5 est mauvais pour les données déséquilibrées.
    # Nous allons fixer le seuil pour qu'il corresponde à la proportion réelle de 'Oui'
    
    # proportion_de_oui = 0.1853 (soit 18.53%)
    proportion_de_oui = y_test.mean()
    
    # 100 - 18.53 = 81.47. On cherche le 81.47e percentile.
    # C'est la probabilité au-dessus de laquelle se trouvent les 18.53% les plus risqués
    seuil_calibre = np.percentile(y_probabilities, 100 - (proportion_de_oui * 100))

    print(f"\nSeuil de décision par défaut (naïf) : 0.5")
    print(f"Seuil de décision calibré (basé sur {proportion_de_oui:.2%}) : {seuil_calibre:.4f}")
    
    # On convertit la probabilité en décision (0 ou 1) avec le NOUVEAU seuil
    y_pred = (y_probabilities > seuil_calibre).astype(int)
    # --- FIN DE LA CORRECTION ---


    # 7. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probabilities) # AUC utilise les probas !
    
    print("\n--- 📊 RÉSULTATS D'ÉVALUATION (sur données Test) ---")
    print(f"Proportion de 'Oui' (sinistre) dans le set de test : {proportion_de_oui:.2%}")
    print("\n--- Métriques de performance (Modèle Calibré) ---")
    print(f"Accuracy  (Précision globale)   : {accuracy:.4f}")
    print(f"AUC       (Capacité à classer)  : {auc:.4f}  <-- (Ne changera pas)")
    print(f"Precision (Sur 'Oui' prédits, % corrects) : {precision:.4f}  <-- (Va s'améliorer)")
    print(f"Recall    (Sur 'Oui' réels, % trouvés)    : {recall:.4f}  <-- (Va s'améliorer)")
    print("\n--- Matrice de Confusion (Modèle Calibré) ---")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------------------------")


if __name__ == "__main__":
    evaluate_probability_model()
