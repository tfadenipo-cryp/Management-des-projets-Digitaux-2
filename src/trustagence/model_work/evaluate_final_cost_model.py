"""
Evaluate Final Cost Model (GLM Tweedie)

This script:
1.  Loads all data (from engineering).
2.  Applies preprocessing and splits into Train (80%) / Test (20%).
3.  Filters for the 13 "stable" features (NOW WITH HISTORY).
4.  Trains the GLM Tweedie model on the Train set.
5.  Makes predictions on the Test set.
6.  Calculates and prints the RMSE and R-squared on unseen data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

# --- Add Project Root to sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# --- End of sys.path modification ---

try:
    from trustagence.engineering.engineering import engineering
    from trustagence.model_work.preprocessing_premium import (
        preprocess_data_for_modeling,
    )
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    sys.exit(1)

# --- CORRECTION : Ajout des features de risque ---
FINAL_FEATURES = [
    "vehicle_age",
    "value_vehicle",
    "payment_0",
    "second_driver_0",
    "driving_experience",
    "type_fuel_0",
    "policies_in_force",
    "lapse",
    "area_0",
    "seniority",
    "distribution_channel_0",
    "n_claims_history",
    "r_claims_history",  # <-- LES AJOUTS IMPORTANTS
]


def evaluate_model() -> None:
    """
    Trains and evaluates the final GLM Tweedie cost model.
    """
    print("--- Starting Final COST Model Evaluation ---")

    # 1. Load data
    print("Step 1: Loading base data...")
    base_df = engineering()
    if base_df is None or base_df.empty:
        print("❌ Failed to load base data.")
        return

    # 2. Preprocess 100% of data and split
    print("Step 2: Preprocessing and splitting data (80% Train / 20% Test)...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        all_feature_names,
    ) = preprocess_data_for_modeling(
        base_df, target_column="cost_claims_year", test_size=0.2, random_state=42
    )

    if X_train is None or X_train.empty:
        print("❌ Preprocessing failed.")
        return

    # 3. Assign correct feature names
    X_train.columns = all_feature_names
    X_test.columns = all_feature_names

    print(f"Step 3: Filtering for {len(FINAL_FEATURES)} selected features...")

    # Handle missing features
    for f in FINAL_FEATURES:
        if f not in X_train.columns:
            print(f"⚠️ Warning: Adding missing feature '{f}' as 0.")
            X_train[f] = 0
            X_test[f] = 0

    X_train_final = X_train[FINAL_FEATURES]
    X_test_final = X_test[FINAL_FEATURES]

    X_train_with_const = sm.add_constant(X_train_final, prepend=True)
    X_test_with_const = sm.add_constant(X_test_final, prepend=True)

    # 4. Train final GLM (Tweedie) model ON TRAINING DATA ONLY
    print(f"Step 4: Training GLM (Tweedie) model on {len(X_train_with_const)} rows...")
    model = sm.GLM(
        y_train,
        X_train_with_const,
        family=sm.families.Tweedie(link=sm.families.links.log()),
    )
    results = model.fit()

    print("✅ Model training complete.")

    # 5. Make predictions ON TEST DATA
    print(
        f"\nStep 5: Making predictions on {len(X_test_with_const)} unseen test rows..."
    )
    # .predict() applique automatiquement l'exponentielle (l'inverse du log)
    y_pred = results.predict(X_test_with_const)

    # 6. Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mean_cost = y_test.mean()

    print("\n--- 📊 RÉSULTATS D'ÉVALUATION (sur données Test) ---")
    print(f"Coût moyen réel (sur set de test) : {mean_cost:,.2f} €")
    print(f"RMSE (Marge d'erreur moyenne)     : {rmse:,.2f} €")
    print(f"R-squared (Performance du modèle) : {r2:,.4f}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    evaluate_model()
