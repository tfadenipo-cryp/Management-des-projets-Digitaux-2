"""
Streamlit Page: Premium Predictor
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

# ------------------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[2]
SRC_DIR = ROOT_DIR / "src"

# ------------------------------------------------------------------------------
# Name stems produced by train script
# ------------------------------------------------------------------------------
MODEL_STEM = "premium_model"            # ✅ Matches your training script
PREPROCESSOR_STEM = "premium_preprocessor"

EXTS = [".pkl", ".pickle", ".joblib"]

# ------------------------------------------------------------------------------
# Features chosen during BIC selection
# ------------------------------------------------------------------------------
FEATURES_LIST = [
    "n_doors_0", "vehicle_age", "value_vehicle", "payment_0", "type_risk_4",
    "second_driver_0", "driving_experience", "type_fuel_0",
    "policies_in_force", "lapse", "area_0", "seniority",
    "distribution_channel_0", "type_risk_1",
]


# ------------------------------------------------------------------------------
# Helper: Find best candidate file
# ------------------------------------------------------------------------------
def _candidate_paths(stem: str, extensions=EXTS):
    env = os.getenv(f"{stem.upper()}_PATH")
    if env:
        p = Path(env)
        yield p if p.is_absolute() else ROOT_DIR / p

    folders = [SRC_DIR / "models", ROOT_DIR / "models"]
    for folder in folders:
        for ext in extensions:
            yield folder / f"{stem}{ext}"


def _resolve_first(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def _load_artifact(path: Path):
    last_err = None
    try:
        return sm.load(path)       # Statsmodels
    except Exception as e:
        last_err = e
    try:
        return joblib.load(path)   # Joblib
    except Exception as e:
        last_err = e
    try:
        with open(path, "rb") as f:
            return pickle.load(f)  # Pickle
    except Exception as e:
        last_err = e
    raise last_err


# ------------------------------------------------------------------------------
# Cached loader
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    model_path = _resolve_first(_candidate_paths(MODEL_STEM))
    prep_path = _resolve_first(_candidate_paths(PREPROCESSOR_STEM))

    model = None
    preprocessor = None

    # Preprocessor
    if prep_path:
        try:
            preprocessor = joblib.load(prep_path)
        except Exception as e:
            st.error(f"❌ Preprocessor could not be loaded at {prep_path}\n{e}")
    else:
        st.error("❌ Preprocessor not found.")

    # Model
    if model_path:
        try:
            model = _load_artifact(model_path)
        except Exception as e:
            st.error(f"❌ Model could not be loaded at {model_path}\n{e}")
    else:
        st.error("❌ Model not found.")

    return preprocessor, model, prep_path, model_path


# ------------------------------------------------------------------------------
# Main Streamlit Page
# ------------------------------------------------------------------------------
def premium_predictor():

    preprocessor, model, prep_path, model_path = load_models()

    st.title("💰 Premium Price Predictor")
    st.caption("Estimate the annual premium based on user attributes.")

    with st.sidebar:
        st.write("Preprocessor:", prep_path)
        st.write("Model:", model_path)

    if not preprocessor or not model:
        st.warning("⚠️ Required model artifacts missing.")
        return

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------
    with st.form("prediction_form"):

        st.subheader("Vehicle")
        col1, col2 = st.columns(2)
        with col1:
            value_vehicle = st.number_input("Vehicle Value (€)", 500, 100000, 20000)
            vehicle_age = st.slider("Vehicle Age (years)", 0, 40, 5)
            type_risk = st.selectbox(
                "Vehicle Type",
                [1, 2, 3, 4],
                format_func=lambda x: {1: "Motorbike", 2: "Van", 3: "Passenger Car", 4: "Agricultural"}[x],
            )
        with col2:
            power = st.number_input("Power (hp)", 40, 500, 110)
            type_fuel = st.selectbox(
                "Fuel Type",
                [1, 2, 0],
                format_func=lambda x: {1: "Petrol", 2: "Diesel", 0: "Other"}[x],
            )
            n_doors = st.selectbox("Number of Doors", [0, 2, 3, 4, 5, 6])

        st.subheader("Driver / Policy")
        col3, col4 = st.columns(2)
        with col3:
            driver_age = st.slider("Driver Age", 18, 90, 35)
            driving_experience = st.slider("Driving Experience (years)", 0, 72, 15)
            area = st.radio("Area", [0, 1], format_func=lambda x: {0: "Rural", 1: "Urban"}[x])
        with col4:
            seniority = st.slider("Seniority (yrs)", 0, 50, 3)
            lapse = st.radio("Policy Lapsed Before?", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
            second_driver = st.radio("Second Driver?", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])

        submit = st.form_submit_button("Predict")

    if not submit:
        return

    # ----------------------------------------------------------------------
    #   Build input DF
    # ----------------------------------------------------------------------
    df = pd.DataFrame({
        "seniority": [seniority],
        "policies_in_force": [1],
        "max_policies": [1],
        "max_products": [1],
        "lapse": [lapse],
        "power": [power],
        "cylinder_capacity": [1600],
        "value_vehicle": [value_vehicle],
        "length": [4.5],
        "weight": [1300],
        "driver_age": [driver_age],
        "vehicle_age": [vehicle_age],
        "driving_experience": [driving_experience],
        "distribution_channel": [0],
        "payment": [0],
        "type_risk": [type_risk],
        "area": [area],
        "second_driver": [second_driver],
        "n_doors": [n_doors],
        "type_fuel": [type_fuel],
    })

    # ----------------------------------------------------------------------
    #   Preprocess
    # ----------------------------------------------------------------------
    X = preprocessor.transform(df)

    num = preprocessor.named_transformers_["num"].feature_names_in_
    cat = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out()
    cols = list(num) + list(cat)

    X_df = pd.DataFrame(X, columns=cols)

    # Ensure required model feature columns exist
    for f in FEATURES_LIST:
        if f not in X_df.columns:
            X_df[f] = 0

    X_df = X_df[FEATURES_LIST]   # selection + ordering (14)

    # ----------------------------------------------------------------------
    #   ADD CONSTANT + ALIGN COLUMN ORDER EXACTLY WITH MODEL
    # ----------------------------------------------------------------------
    X_df_sm = sm.add_constant(X_df, prepend=True, has_constant="add")

    expected_cols = list(model.model.exog_names)

    # Add missing columns, if any
    for c in expected_cols:
        if c not in X_df_sm.columns:
            X_df_sm[c] = 0.0

    # Sort columns exactly in same order as model expects
    X_df_sm = X_df_sm[expected_cols]

    # ----------------------------------------------------------------------
    #   Predict
    # ----------------------------------------------------------------------
    y_log = model.predict(X_df_sm)
    premium = float(np.exp(y_log)[0])   # GLM(Gamma(log)) → exp backtransform

    st.success(f"## ✅ Estimated Premium: €{premium:,.2f}")

    with st.expander("Debug - Model Input (X)"):
        st.write("Expected cols:", expected_cols)
        st.write(X_df_sm)



