"""
Train Final Premium Model

This script:
1.  Loads all data (from engineering).
2.  Preprocesses 100% of the data (using preprocessing.py).
3.  Filters for the 14 "winning" features from the BIC analysis.
4.  Trains the final GLM (Gamma) model on ALL data.
5.  Saves the fitted preprocessor and the trained model to disk.
"""
import pandas as pd
import statsmodels.api as sm
import sys
import joblib # Used for saving model files
from pathlib import Path
import numpy as np # <-- IMPORTED NUMPY

# --- Add Project Root to sys.path to allow for src imports ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# --- End of sys.path modification ---

try:
    from src.functions.engineering import engineering
    from src.models.preprocessing import preprocess_data_for_modeling
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure all __init__.py files are present.")
    print(f"Details: {e}")
    sys.exit(1)

# --- THESE ARE THE 14 WINNING FEATURES FROM YOUR BIC ANALYSIS ---
FINAL_FEATURES = [
    'n_doors_0', 'vehicle_age', 'value_vehicle', 'payment_0', 'type_risk_4', 
    'second_driver_0', 'driving_experience', 'type_fuel_0', 
    'policies_in_force', 'lapse', 'area_0', 'seniority', 
    'distribution_channel_0', 'type_risk_1'
]

# --- FILE PATHS FOR SAVED MODELS ---
PREPROCESSOR_PATH = ROOT_DIR / "models/premium_preprocessor.joblib"
MODEL_PATH = ROOT_DIR / "models/premium_model.pkl"

def train_and_save_model():
    print("--- Starting Final Model Training ---")
    
    # 1. Load base data
    print("Step 1: Loading base data...")
    base_df = engineering()
    if base_df is None or base_df.empty:
        print("❌ Failed to load base data.")
        return

    # 2. Preprocess 100% of the data
    print("Step 2: Preprocessing 100% of data...")
    # We get 6 items back, including 'all_features'
    X_train, X_test, y_train, y_test, preprocessor, all_features = \
        preprocess_data_for_modeling(base_df, target_column='premium', test_size=0.01)

    if X_train is None or X_train.empty:
        print("❌ Preprocessing failed.")
        return
        
    # Combine train and test to get the full 100% dataset
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    # 3. Filter for our 14 "winning" features
    print(f"Step 3: Filtering for {len(FINAL_FEATURES)} selected features...")
    
    # --- THIS IS THE FIX for the NameError ---
    # The returned variable is 'all_features', not 'all_feature_names'
    X_full.columns = all_features
    # --- END OF FIX ---
    
    # Check for missing features
    missing_features = [f for f in FINAL_FEATURES if f not in X_full.columns]
    if missing_features:
        print(f"❌ Error: The following required features are missing from the preprocessed data: {missing_features}")
        # This can happen if a category (e.g., n_doors_0) wasn't in the 1% test split
        # For simplicity, we'll continue, but a robust pipeline would handle this
        print("Attempting to continue by adding missing features as 0...")
        for f in missing_features:
            X_full[f] = 0 # Add missing one-hot columns as 0
            
    X_final = X_full[FINAL_FEATURES]
    X_final_with_const = sm.add_constant(X_final, prepend=True)

    # 4. Train the final GLM Gamma model
    print(f"Step 4: Training final GLM (Gamma) model on {len(X_final)} rows...")
    final_model = sm.GLM(y_full, X_final_with_const, 
                         family=sm.families.Gamma(link=sm.families.links.log()))
    final_model_results = final_model.fit()

    print("✅ Model training complete.")
    print(final_model_results.summary())

    # 5. Save the preprocessor and the model
    print("\nStep 5: Saving model and preprocessor to disk...")
    
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"✅ Preprocessor saved to: {PREPROCESSOR_PATH}")
    
    final_model_results.save(MODEL_PATH)
    print(f"✅ Final model saved to: {MODEL_PATH}")
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    train_and_save_model()