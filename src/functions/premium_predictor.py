"""
Streamlit Page: Premium Predictor
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import statsmodels.api as sm
import streamlit as st

# ------------------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[2]
MODELS_DIR = ROOT_DIR / "models"

PREPROCESSOR_PATH = MODELS_DIR / "premium_preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "premium_model.pkl"
FEATURES_PATH = MODELS_DIR / "premium_model_features.json"

# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
@st.cache_resource
def load_premium_models() -> tuple:
    """
    Loads the preprocessor, trained GLM model, and the ordered feature list.
    """
    preprocessor, model, features = None, None, None

    # ---- Load Preprocessor ----
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        st.error(f"❌ Could not load preprocessor:\n{e}")

    # ---- Load Model ----
    try:
        model = sm.load(MODEL_PATH)
    except Exception:
        try:
            model = sm.load_results(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Could not load model:\n{e}")

    # ---- Load Feature Names ----
    try:
        with open(FEATURES_PATH, "r") as f:
            features = json.load(f)
    except Exception as e:
        st.error(f"❌ Could not load model feature names:\n{e}")

    return preprocessor, model, features


# ------------------------------------------------------------------------------
# Main Streamlit Page
# ------------------------------------------------------------------------------
def premium_predictor() -> None:
    """
    Main function for the Premium Predictor page.
    """
    preprocessor, model, expected_features = load_premium_models()

    # UI Title
    st.title("Premium Price Predictor")
    st.markdown("Enter the details below to get an estimated yearly insurance premium.")

    if not all([preprocessor, model, expected_features]):
        st.warning(
            "⚠️ One or more model artifacts could not be found.\n"
            "Please run: `uv run python src/models/final_model_premium.py`"
        )
        return

    # ----------------------------------------------------------------------
    # Form Input
    # ----------------------------------------------------------------------
    with st.form(key="prediction_form_premium"):

        st.subheader("Vehicle Information")
        col1, col2 = st.columns(2)
        with col1:
            value_vehicle = st.number_input(
                "Vehicle Value (€)", min_value=500, max_value=100000, value=20000, step=500
            )
            vehicle_age = st.slider("Vehicle Age (years)", min_value=0, max_value=40, value=5)
            type_risk = st.selectbox(
                "Vehicle Type",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Motorbike",
                    2: "Van",
                    3: "Passenger Car",
                    4: "Agricultural",
                }.get(x),
            )
        with col2:
            power = st.number_input("Power (hp)", min_value=40, max_value=500, value=110)
            type_fuel = st.selectbox(
                "Fuel Type",
                options=[1, 2, 0],
                format_func=lambda x: {
                    1: "Petrol",
                    2: "Diesel",
                    0: "Other",
                }.get(x),
            )
            n_doors = st.selectbox(
                "Number of Doors",
                options=[0, 2, 3, 4, 5, 6],
                format_func=lambda x: f"{x} doors" if x > 0 else "N/A (e.g. Motorbike)",
            )

        st.subheader("Driver & Policy Details")
        col3, col4 = st.columns(2)
        with col3:
            driver_age = st.slider("Driver Age", min_value=18, max_value=90, value=35)
            driving_experience = st.slider(
                "Driving Experience (years)", min_value=0, max_value=72, value=15
            )
            area = st.radio(
                "Area",
                options=[0, 1],
                format_func=lambda x: {0: "Rural", 1: "Urban"}.get(x),
            )
        with col4:
            seniority = st.slider("Seniority (years)", min_value=0, max_value=50, value=3)
            lapse = st.radio(
                "Policy Lapsed Before?",
                options=[0, 1],
                format_func=lambda x: {0: "No", 1: "Yes"}.get(x),
            )
            second_driver = st.radio(
                "Has Second Driver?",
                options=[0, 1],
                format_func=lambda x: {0: "No", 1: "Yes"}.get(x),
            )

        # Fixed dummy variables
        policies_in_force = 1
        payment = 0
        distribution_channel = 0

        submit_button = st.form_submit_button(label="Predict Premium")

    # ----------------------------------------------------------------------
    # Prediction Logic
    # ----------------------------------------------------------------------
    if not submit_button:
        return

    # Step 1: Build DF
    input_data = pd.DataFrame({
        "seniority": [seniority],
        "policies_in_force": [policies_in_force],
        "max_policies": [policies_in_force],
        "max_products": [1],
        "lapse": [lapse],
        "power": [power],
        "cylinder_capacity": [1600],  # Dummy
        "value_vehicle": [value_vehicle],
        "length": [4.5],   # Dummy
        "weight": [1300],  # Dummy
        "driver_age": [driver_age],
        "vehicle_age": [vehicle_age],
        "driving_experience": [driving_experience],
        "distribution_channel": [distribution_channel],
        "payment": [payment],
        "type_risk": [type_risk],
        "area": [area],
        "second_driver": [second_driver],
        "n_doors": [n_doors],
        "type_fuel": [type_fuel],
    })

    # Step 2: Preprocess
    input_processed = preprocessor.transform(input_data)

    # Step 3: Rebuild column names
    num_features = preprocessor.named_transformers_["num"].feature_names_in_
    cat_features = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out()
    all_feature_names = list(num_features) + list(cat_features)

    input_df = pd.DataFrame(input_processed, columns=all_feature_names)

    # Step 4: Add constant + feature alignment
    input_df_const = sm.add_constant(input_df, prepend=True, has_constant="add")

    aligned_input = pd.DataFrame(columns=expected_features, index=[0]).fillna(0.0)

    for col in aligned_input.columns:
        if col in input_df_const.columns:
            aligned_input[col] = input_df_const[col].values

    # Step 5: Prediction
    predicted_premium = model.predict(aligned_input)
    premium = float(predicted_premium[0])

    st.success(f"## ✅ Estimated Annual Premium: €{premium:,.2f}")

    with st.expander("Debug - Processed Model Input"):
        st.write("Model Expected Columns:", expected_features)
        st.write(aligned_input)
