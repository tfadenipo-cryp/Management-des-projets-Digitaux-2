from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore
import pandas as pd
import statsmodels.api as sm  # type: ignore
import streamlit as st


# --- Define paths ---
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[3]
MODELS_DIR = ROOT_DIR / "src" / "trustagence" / "models_docs"
SRC_DIR = ROOT_DIR / "src"
PREPROCESSOR_PATH = MODELS_DIR / "premium_preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "premium_model.pkl"
FEATURES_PATH = MODELS_DIR / "premium_model_features.json"


# --- Model Loading ---
@st.cache_resource
def load_premium_models() -> tuple:
    """
    Loads the preprocessor, the trained GLM model, and feature list.
    """
    preprocessor = None
    model = None
    features = []

    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        st.error(f"Preprocessor could not be loaded: {e}")

    try:
        model = sm.load(MODEL_PATH)
    except Exception:
        try:
            model = sm.load_results(MODEL_PATH)
        except Exception as e:
            st.error(f"Model could not be loaded: {e}")

    try:
        features = pd.read_json(FEATURES_PATH, typ="series").to_list()
    except Exception as e:
        st.error(f"Features file could not be loaded: {e}")

    return preprocessor, model, features


def premium_predictor() -> None:
    """
    Main function for the Premium Predictor page.
    """
    preprocessor, model, expected_features = load_premium_models()

    # Page Title
    st.title("Insurance Premium Predictor")
    st.markdown("Enter your details to get an estimate of your annual premium.")

    # User Input Form
    if not all([preprocessor, model, expected_features]):
        st.warning(
            "One or more model artifacts are missing."
            "Please run `uv run python src/models/final_model_premium.py`"
        )
        return

    with st.form(key="prediction_form_premium"):
        st.subheader("Vehicle Details")
        col1, col2 = st.columns(2)
        with col1:
            value_vehicle = st.number_input(
                "Vehicle Value (€)",
                min_value=500,
                max_value=100000,
                value=20000,
                step=500,
            )
            vehicle_age = st.slider(
                "Vehicle Age (years)", min_value=0, max_value=40, value=5
            )

            # --- CORRECTION 1 ---
            type_risk = st.selectbox(
                "Vehicle Type",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Motorcycle",
                    2: "Van",
                    3: "Car",
                    4: "Agricultural",
                }.get(x, str(x)),
            )
            # --- END CORRECTION 1 ---

        with col2:
            power = st.number_input(
                "Horsepower (hp)", min_value=40, max_value=500, value=110
            )

            # --- CORRECTION 2 ---
            type_fuel = st.selectbox(
                "Fuel Type",
                options=[1, 2, 0],
                format_func=lambda x: {1: "Petrol", 2: "Diesel", 0: "Other"}.get(
                    x, str(x)
                ),
            )
            # --- END CORRECTION 2 ---

            n_doors = st.selectbox(
                "Number of Doors",
                options=[0, 2, 3, 4, 5, 6],
                format_func=lambda x: f"{x} doors"
                if x > 0
                else "N/A (e.g., Motorcycle)",
            )

        st.subheader("Driver & Contract Details")
        col3, col4 = st.columns(2)
        with col3:
            driver_age = st.slider("Driver Age", min_value=18, max_value=90, value=35)
            driving_experience = st.slider(
                "Years of Experience", min_value=0, max_value=72, value=15
            )
            area = st.radio(
                "Area",
                options=[0, 1],
                format_func=lambda x: {0: "Rural", 1: "Urban"}.get(x),
            )
        with col4:
            seniority = st.slider(
                "Seniority (years)", min_value=0, max_value=50, value=3
            )
            lapse = st.radio(
                "Contract previously cancelled?",
                options=[0, 1],
                format_func=lambda x: {0: "No", 1: "Yes"}.get(x),
            )
            second_driver = st.radio(
                "Second Driver?",
                options=[0, 1],
                format_func=lambda x: {0: "No", 1: "Yes"}.get(x),
            )

        # Fixed variables required by preprocessor
        policies_in_force = 1
        payment = 0
        distribution_channel = 0

        submit_button = st.form_submit_button(label="Estimate My Premium")

    # --- Prediction Logic ---
    if submit_button:
        input_data = pd.DataFrame(
            {
                "seniority": [seniority],
                "policies_in_force": [policies_in_force],
                "max_policies": [policies_in_force],
                "max_products": [1],
                "lapse": [lapse],
                "power": [power],
                "cylinder_capacity": [1600],  # Dummy
                "value_vehicle": [value_vehicle],
                "length": [4.5],  # Dummy
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
            }
        )

        # Step 1: Transform data
        input_processed = preprocessor.transform(input_data)

        # Step 2: Recreate DataFrame with proper column names
        num_features = preprocessor.named_transformers_["num"].feature_names_in_
        cat_features = preprocessor.named_transformers_["cat"][
            "onehot"
        ].get_feature_names_out()
        all_feature_names = list(num_features) + list(cat_features)
        input_df = pd.DataFrame(input_processed, columns=all_feature_names)

        # Step 3: Add constant
        input_df_with_const = sm.add_constant(
            input_df, prepend=True, has_constant="add"
        )

        # Step 4: Align columns (CRUCIAL)
        input_aligned = pd.DataFrame(columns=expected_features, index=[0]).fillna(0.0)

        for col in input_aligned.columns:
            if col in input_df_with_const.columns:
                input_aligned[col] = input_df_with_const[col].values

        # Step 5: Prediction
        predicted_premium = model.predict(input_aligned)
        premium = float(predicted_premium[0])

        # Display result
        st.success(f"Estimated Annual Premium: €{premium:,.2f}")
