"""
Streamlit Page: Probability Predictor (for Insurer)
(Filename cost_predictor.py is kept for import compatibility)
"""

from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore
import pandas as pd
import statsmodels.api as sm  # type: ignore
import streamlit as st

# --- Project paths ---
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[3]
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"

# --- Artifact Stems ---
# We load the "cost" files which now contain
# the PROBABILITY model
MODEL_STEM = "cost_model"
PREPROCESSOR_STEM = "cost_preprocessor"
FEATURES_STEM = "cost_model_features"


@st.cache_resource
def load_cost_models() -> tuple:
    """
    Loads the probability model, preprocessor, and feature list.
    """
    model_path = MODELS_DIR / f"{MODEL_STEM}.pkl"
    prep_path = MODELS_DIR / f"{PREPROCESSOR_STEM}.joblib"
    features_path = MODELS_DIR / f"{FEATURES_STEM}.json"

    model = None
    preprocessor = None
    features = []

    # Load Preprocessor
    if prep_path.exists():
        try:
            preprocessor = joblib.load(prep_path)
        except Exception as e:
            st.error(f"❌ Preprocessor could not be loaded: {e}")
    else:
        st.error(f"❌ Preprocessor not found at {prep_path}")

    # Load Model
    if model_path.exists():
        try:
            model = sm.load(model_path)
        except Exception as e:
            st.error(f"❌ Model could not be loaded: {e}")
    else:
        st.error(f"❌ Model not found at {model_path}")

    # Load Features
    if features_path.exists():
        try:
            features = pd.read_json(features_path, typ="series").to_list()
        except Exception as e:
            st.error(f"❌ Features file could not be loaded: {e}")
    else:
        st.error(f"❌ Features file not found at {features_path}")

    return preprocessor, model, features


def cost_predictor() -> None:
    """
    Streamlit page for predicting the PROBABILITY of a claim.
    """
    preprocessor, model, expected_features = load_cost_models()

    # --- CORRECTION: Title updated ---
    st.title("Claim Probability Predictor")
    st.markdown(
        "Estimate the **probability that a profile will have a claim** during the year."
    )

    if not all([preprocessor, model, expected_features]):
        st.warning(
            "⚠️ One or more model artifacts are missing."
            "Please run `uv run python src/models/final_model_cost.py`"
        )
        return

    # --- Prediction Form ---
    with st.form("prediction_form_cost"):
        st.subheader("Vehicle")
        col1, col2 = st.columns(2)
        with col1:
            value_vehicle = st.number_input("Vehicle Value (€)", 500, 100000, 20000)
            vehicle_age = st.slider("Vehicle Age (years)", 0, 40, 5)
            type_risk = st.selectbox(
                "Vehicle Type",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Motorcycle",
                    2: "Van",
                    3: "Car",
                    4: "Agricultural",
                }[x],
            )
        with col2:
            power = st.number_input("Power (hp)", 40, 500, 110)
            type_fuel = st.selectbox(
                "Fuel Type",
                [1, 2, 0],
                format_func=lambda x: {1: "Gasoline", 2: "Diesel", 0: "Other"}[x],
            )
            n_doors = st.selectbox("Number of Doors", [0, 2, 3, 4, 5, 6])

        st.subheader("Driver / Contract")
        col3, col4 = st.columns(2)
        with col3:
            driver_age = st.slider("Driver Age", 18, 90, 35)
            driving_experience = st.slider("Years of Experience", 0, 72, 15)
            area = st.radio(
                "Area", [0, 1], format_func=lambda x: {0: "Rural", 1: "Urban"}[x]
            )
        with col4:
            seniority = st.slider("Seniority (years)", 0, 50, 3)
            lapse = st.radio(
                "Contract previously terminated?",
                [0, 1],
                format_func=lambda x: {0: "No", 1: "Yes"}[x],
            )
            second_driver = st.radio(
                "Second driver?", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x]
            )

        submit = st.form_submit_button("Estimate Risk Probability")

    if not submit:
        return

    # --- Build Input DataFrame ---
    # These values are placeholders for features not in the form
    df = pd.DataFrame(
        {
            "value_vehicle": [value_vehicle],
            "vehicle_age": [vehicle_age],
            "type_risk": [type_risk],
            "power": [power],
            "type_fuel": [type_fuel],
            "n_doors": [n_doors],
            "driver_age": [driver_age],
            "driving_experience": [driving_experience],
            "area": [area],
            "seniority": [seniority],
            "lapse": [lapse],
            "second_driver": [second_driver],
            "policies_in_force": [1],  # Placeholder
            "max_policies": [1],  # Placeholder
            "max_products": [1],  # Placeholder
            "cylinder_capacity": [1600],  # Placeholder
            "length": [4.5],  # Placeholder
            "weight": [1300],  # Placeholder
            "distribution_channel": [0],  # Placeholder
            "payment": [0],  # Placeholder
        }
    )

    # --- Preprocess ---
    X_processed = preprocessor.transform(df)

    num_features = preprocessor.named_transformers_["num"].feature_names_in_
    cat_features = preprocessor.named_transformers_["cat"][
        "onehot"
    ].get_feature_names_out()
    all_processed_features = list(num_features) + list(cat_features)

    X_df = pd.DataFrame(X_processed, columns=all_processed_features)
    X_df_sm = sm.add_constant(X_df, prepend=True, has_constant="add")

    # Align columns with the model's expected features
    X_final = pd.DataFrame(columns=expected_features, index=[0]).fillna(0.0)

    for col in X_final.columns:
        if col in X_df_sm.columns:
            X_final[col] = X_df_sm[col].values

    # --- Predict ---
    try:
        probability = model.predict(X_final)
        prob_percent = probability[0] * 100

        st.success(f"## ✅ Claim Probability: {prob_percent:,.1f} %")

        if prob_percent < 15:
            st.info("Low Risk")
        elif prob_percent < 30:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
