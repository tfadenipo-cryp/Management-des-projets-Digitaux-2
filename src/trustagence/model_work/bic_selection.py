"""
BIC Model Selection Script
This script performs forward stepwise feature selection using
BIC (Bayesian Information Criterion) to find an optimal model
for predicting insurer cost (cost_claims_year) and calculates
its prediction error.
"""

import pandas as pd
import statsmodels.api as sm
import sys
import numpy as np
from pathlib import Path


try:
    from sklearn.metrics import mean_squared_error
except ImportError:
    print(
        "Error: Could not import sklearn.metrics. Please ensure scikit-learn is installed."
    )
    sys.exit(1)


ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from trustagence.engineering.engineering import engineering
    from trustagence.model_work.preprocessing import preprocess_data_for_modeling
except ImportError as e:
    print(
        "Error: Could not import necessary modules. Make sure all __init__.py files are present."
    )
    print(f"Details: {e}")
    sys.exit(1)


def perform_bic_selection(X, y):
    """
    Performs forward stepwise selection using BIC on a GLM.
    ... (rest of the docstring) ...
    """
    print("\n--- Starting BIC Forward Stepwise Selection ---")

    included_features = []
    all_features = list(X.columns)

    X_const = sm.add_constant(pd.DataFrame(index=X.index), prepend=True)
    try:
        baseline_model = sm.GLM(
            y, X_const, family=sm.families.Tweedie(link=sm.families.links.log())
        ).fit()
        current_bic = baseline_model.bic
        best_model_results = baseline_model
        print(f"Baseline (Intercept-only) BIC: {current_bic:,.2f}")
    except Exception as e:
        print(f"Error fitting baseline model: {e}")
        return None, None

    while True:
        best_bic_this_step = current_bic
        feature_to_add = None
        best_model_results_this_step = None

        print("\n--- Testing remaining features ---")

        for feature in all_features:
            if feature in included_features:
                continue

            potential_features = included_features + [feature]
            X_step = X[potential_features]
            X_step = sm.add_constant(X_step, prepend=True)

            try:
                model = sm.GLM(
                    y, X_step, family=sm.families.Tweedie(link=sm.families.links.log())
                )
                results = model.fit()
                step_bic = results.bic

                if step_bic < best_bic_this_step:
                    best_bic_this_step = step_bic
                    feature_to_add = feature
                    best_model_results_this_step = results

            except Exception:
                continue

        if feature_to_add:
            current_bic = best_bic_this_step
            included_features.append(feature_to_add)
            best_model_results = best_model_results_this_step
            print(f"✅ ADDED: '{feature_to_add}' | New Model BIC: {current_bic:,.2f}")
        else:
            print("\n--- No feature addition improved BIC. Stopping selection. ---")
            break

    print(f"\nFinal Selected Features: {included_features}")
    return included_features, best_model_results


def main():
    """
    Main function to run the data loading, preprocessing,
    and BIC selection process.
    """
    print("--- Starting Modeling Process ---")

    print("Step 1: Loading base data via engineering()...")
    base_df = engineering()
    if base_df is None or base_df.empty:
        print("❌ Failed to load base data. Exiting.")
        return

    print("Step 2: Applying shared preprocessing...")

    X_train, X_test, y_train, y_test = preprocess_data_for_modeling(
        base_df, target_column="cost_claims_year"
    )

    if X_train is None or X_train.empty or y_train.empty:
        print("❌ Preprocessing failed or resulted in empty data. Exiting.")
        return

    # BIC
    print("Step 3: Running BIC stepwise selection...")
    selected_features, final_model_results = perform_bic_selection(X_train, y_train)

    if final_model_results:
        print("\n--- Final Model Summary (based on BIC) ---")
        print(final_model_results.summary())

        print("\n--- Prediction Error (on Test Data) ---")

        X_test_selected = X_test[selected_features]
        X_test_with_const = sm.add_constant(X_test_selected, prepend=True)

        try:
            y_pred = final_model_results.predict(X_test_with_const)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            print(f"Mean Squared Error (MSE): {mse:,.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
            print(f"Mean cost_claims_year (for comparison): {y_test.mean():,.2f}")

        except Exception as e:
            print(f"Error during prediction on test set: {e}")

    else:
        print("BIC selection did not find a final model.")


if __name__ == "__main__":
    main()
