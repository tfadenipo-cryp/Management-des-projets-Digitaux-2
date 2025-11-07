"""
Train Final PROBABILITY Model (for Insurer/Décideur) - VERSION CORRIGÉE

This script:
1.  Loads all data, creates 'had_claim' target, and REMOVES history columns.
2.  Calls the preprocessor (which will ignore the removed columns).
3.  Filters for the 14 "winning" features.
4.  Trains the final GLM Binomial (Logistic Regression) model on ALL data.
5.  Saves the fitted preprocessor and the trained model to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib  # type: ignore
import pandas as pd
import statsmodels.api as sm  # type: ignore

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
    print(f"Error: Could not import necessary modules. {e}")
    sys.exit(1)

# Les 14 features stables du modèle Premium
FINAL_FEATURES = [
    "n_doors_0",
    "vehicle_age",
    "value_vehicle",
    "payment_0",
    "type_risk_4",
    "second_driver_0",
    "driving_experience",
    "type_fuel_0",
    "policies_in_force",
    "lapse",
    "area_0",
    "seniority",
    "distribution_channel_0",
    "type_risk_1",
]

# Path for saved models (Nous écrasons les anciens fichiers 'cost')
PREPROCESSOR_PATH = ROOT_DIR / "models/cost_preprocessor.joblib"
MODEL_PATH = ROOT_DIR / "models/cost_model.pkl"
ALL_FEATURES_PATH = ROOT_DIR / "models/cost_model_features.json"


def train_and_save_probability_model() -> None:
    """
    Trains and saves the GLM (Binomial) model for 'had_claim'.
    """
    print("--- Starting Final PROBABILITY Model Training ---")

    # 1. Load data
    print("Step 1: Loading base data...")
    base_df = engineering()
    if base_df is None or base_df.empty:
        print("❌ Failed to load base data.")
        return

    # 2. CRÉER LA CIBLE (y) ET NETTOYER LES FEATURES
    print("Step 2: Creating target 'had_claim' and removing leaks...")
    base_df["had_claim"] = (base_df["n_claims_year"] > 0).astype(int)

    # --- CORRECTION DÉFINITIVE ---
    # Nous enlevons les "fuites de données" AVANT d'appeler le préprocesseur.
    # Le préprocesseur ne doit jamais voir les colonnes que l'utilisateur ne peut pas fournir.
    cols_to_remove_manually = [
        "n_claims_history",
        "r_claims_history",
        "cost_claims_year",
        "n_claims_year",
        "premium",  # Le premium n'est pas une feature
    ]
    base_df_cleaned = base_df.drop(columns=cols_to_remove_manually, errors="ignore")
    # --- FIN DE LA CORRECTION ---

    # 3. Preprocess 100% of data
    print("Step 3: Preprocessing 100% of data...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        all_feature_names,
    ) = preprocess_data_for_modeling(
        base_df_cleaned, target_column="had_claim", test_size=0.01, random_state=42
    )

    if X_train is None or X_train.empty:
        print("❌ Preprocessing failed.")
        return

    # Recombine into full dataset for final training
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    # Assign correct feature names
    X_full.columns = all_feature_names

    print(f"Step 4: Filtering for {len(FINAL_FEATURES)} selected features...")
    for f in FINAL_FEATURES:
        if f not in X_full.columns:
            print(f"⚠️ Warning: Adding missing feature '{f}' as 0.")
            X_full[f] = 0

    X_final = X_full[FINAL_FEATURES]
    X_final_with_const = sm.add_constant(X_final, prepend=True)

    # 4. Train final GLM (Binomial / Logistic) model
    print(f"Step 5: Training final GLM (Binomial) model on {len(X_final)} rows...")
    final_model = sm.GLM(
        y_full,
        X_final_with_const,
        family=sm.families.Binomial(link=sm.families.links.logit()),
    )
    final_model_results = final_model.fit()

    print("✅ Model training complete.")
    print(final_model_results.summary())

    # 5. Saving the preprocessor and the model
    print("\nStep 6: Saving model and preprocessor to disk...")
    (ROOT_DIR / "models").mkdir(exist_ok=True)

    # Ce préprocesseur est maintenant "propre"
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"✅ Preprocessor saved to: {PREPROCESSOR_PATH}")

    final_model_results.save(MODEL_PATH)
    print(f"✅ Final model saved to: {MODEL_PATH}")

    # Save feature list and order
    expected_cols_order = X_final_with_const.columns.to_list()
    pd.Series(expected_cols_order).to_json(ALL_FEATURES_PATH)
    print(f"✅ Feature list saved to: {ALL_FEATURES_PATH}")
    print("\n--- Training Complete ---")


if __name__ == "__main__":
    train_and_save_probability_model()
