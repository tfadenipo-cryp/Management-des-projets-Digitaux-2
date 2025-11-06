from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "premium_preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "premium_model.pkl"
FEATURES_PATH = MODELS_DIR / "premium_model_features.json"


def _load_premium_models_raw() -> tuple:
    preprocessor = None
    model = None
    features = []
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception:
        pass
    try:
        model = sm.load(MODEL_PATH)
    except Exception:
        try:
            model = sm.load_results(MODEL_PATH)
        except Exception:
            pass
    try:
        features = pd.read_json(FEATURES_PATH, typ="series").to_list()
    except Exception:
        pass
    return preprocessor, model, features


@st.cache_resource
def load_premium_models() -> tuple:
    return _load_premium_models_raw()


def premium_predictor() -> None:
    preprocessor, model, expected_features = load_premium_models()

    st.title("💰 Prédicteur de Prime d'Assurance")
    st.markdown("Entrez vos détails pour obtenir une estimation de votre prime annuelle.")

    if not all([preprocessor, model, expected_features]):
        st.warning("⚠️ Artefacts de modèle manquants.")
        return

    with st.form(key="prediction_form_premium"):
        st.subheader("Détails du Véhicule")
        col1, col2 = st.columns(2)
        with col1:
            value_vehicle = st.number_input("Valeur du véhicule (€)", min_value=500, max_value=100000, value=20000, step=500)
            vehicle_age = st.slider("Âge du véhicule (années)", min_value=0, max_value=40, value=5)
            type_risk = st.selectbox("Type de véhicule", options=[1, 2, 3, 4])
        with col2:
            power = st.number_input("Puissance (ch)", min_value=40, max_value=500, value=110)
            type_fuel = st.selectbox("Carburant", options=[1, 2, 0])
            n_doors = st.selectbox("Nombre de portes", options=[0, 2, 3, 4, 5, 6])

        st.subheader("Détails Conducteur & Contrat")
        col3, col4 = st.columns(2)
        with col3:
            driver_age = st.slider("Âge du conducteur", min_value=18, max_value=90, value=35)
            driving_experience = st.slider("Années d'expérience", min_value=0, max_value=72, value=15)
            area = st.radio("Zone", options=[0, 1])
        with col4:
            seniority = st.slider("Ancienneté (années)", min_value=0, max_value=50, value=3)
            lapse = st.radio("Contrat déjà résilié ?", options=[0, 1])
            second_driver = st.radio("Second conducteur ?", options=[0, 1])

        policies_in_force = 1
        payment = 0
        distribution_channel = 0

        submit_button = st.form_submit_button(label="Estimer ma Prime")

    if submit_button:
        input_data = pd.DataFrame({
            'seniority': [seniority],
            'policies_in_force': [policies_in_force],
            'max_policies': [policies_in_force],
            'max_products': [1],
            'lapse': [lapse],
            'power': [power],
            'cylinder_capacity': [1600],
            'value_vehicle': [value_vehicle],
            'length': [4.5],
            'weight': [1300],
            'driver_age': [driver_age],
            'vehicle_age': [vehicle_age],
            'driving_experience': [driving_experience],
            'distribution_channel': [distribution_channel],
            'payment': [payment],
            'type_risk': [type_risk],
            'area': [area],
            'second_driver': [second_driver],
            'n_doors': [n_doors],
            'type_fuel': [type_fuel]
        })

        input_processed = preprocessor.transform(input_data)

        num_features = preprocessor.named_transformers_['num'].feature_names_in_
        cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
        all_feature_names = list(num_features) + list(cat_features)
        input_df = pd.DataFrame(input_processed, columns=all_feature_names)
        input_df = sm.add_constant(input_df, prepend=True, has_constant="add")
        input_aligned = pd.DataFrame(columns=expected_features, index=[0]).fillna(0.0)
        for col in input_aligned.columns:
            if col in input_df.columns:
                input_aligned[col] = input_df[col].values

        pred = model.predict(input_aligned)
        premium = float(pred[0])
        st.success(f"## Prime Annuelle Estimée : €{premium:,.2f}")
