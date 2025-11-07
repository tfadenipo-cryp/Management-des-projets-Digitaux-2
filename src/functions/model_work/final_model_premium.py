"""
Train Final Premium Model

This script:
1. Loads all data (from engineering).
2. Preprocesses 100% of the data (using preprocessing.py).
3. Filters for the 14 "winning" features from the BIC analysis.
4. Trains the final GLM model on ALL data.
5. Saves the fitted preprocessor, model, and model feature list to disk.
"""

import pandas as pd
import statsmodels.api as sm
import sys
import joblib
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from src.functions.engineering import engineering
    from src.models.preprocessing import preprocess_data_for_modeling
except ImportError as e:
    print(
        "Error: Could not import necessary modules. Make sure all __init__.py files are present."
    )
    print(f"Details: {e}")
    sys.exit(1)

# 14 WINNING FEATURES FROM THE BIC ANALYSIS
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

# Path for saved models
MODELS_DIR = ROOT_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "premium_preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "premium_model.pkl"
FEATURES_PATH = MODELS_DIR / "premium_model_features.json"


def train_and_save_model():
    print("--- Starting Final Model Training ---")

    print("Step 1: Loading base data...")
    base_df = engineering()
    if base_df is None or base_df.empty:
        print("❌ Failed to load base data.")
        return

    print("Step 2: Preprocessing 100% of data...")
    X_train, X_test, y_train, y_test, preprocessor, all_features = (
        preprocess_data_for_modeling(base_df, target_column="premium", test_size=0.01)
    )

    if X_train is None or X_train.empty:
        print("❌ Preprocessing failed.")
        return

    # reassemble full set after split
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    print(f"Step 3: Filtering for {len(FINAL_FEATURES)} selected features...")

    X_full.columns = all_features

    missing_features = [f for f in FINAL_FEATURES if f not in X_full.columns]
    if missing_features:
        print(f"⚠️ Missing features detected: {missing_features}")
        print("→ Adding them as zeros to allow model to train...")
        for f in missing_features:
            X_full[f] = 0

    X_final = X_full[FINAL_FEATURES]
    X_final_const = sm.add_constant(X_final, prepend=True)

    # Final GLM
    print(f"Step 4: Training final GLM (Gamma) model on {len(X_final)} rows...")
    final_model = sm.GLM(
        y_full, X_final_const, family=sm.families.Gamma(link=sm.families.links.log())
    )
    final_model_results = final_model.fit()

    print("✅ Model training complete.")
    print(final_model_results.summary())

    # Saving preprocessor + model
    MODELS_DIR.mkdir(exist_ok=True)

    print("\nStep 5: Saving model + preprocessor...")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"✅ Saved preprocessor → {PREPROCESSOR_PATH}")

    final_model_results.save(MODEL_PATH)
    print(f"✅ Saved model → {MODEL_PATH}")

    # -------------------------------------------------------
    # ✅ SAVE MODEL EXOG NAMES → REQUIRED FOR STREAMLIT
    # -------------------------------------------------------
    try:
        with open(FEATURES_PATH, "w") as f:
            json.dump(final_model_results.model.exog_names, f, indent=2)
        print(f"✅ Saved model feature names → {FEATURES_PATH}")
    except Exception as e:
        print(f"❌ Could not save model features JSON: {e}")

    print("\n--- Training Complete ✅ ---")


if __name__ == "__main__":
    train_and_save_model()
