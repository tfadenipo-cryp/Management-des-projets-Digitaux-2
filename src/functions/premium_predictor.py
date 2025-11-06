"""
Streamlit Page: Premium Predictor
"""
from __future__ import annotations

from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
import streamlit as st

# --- Define paths ---
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
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
        st.error(f"❌ Preprocessor could not be loaded: {e}")

    try:
        model = sm.load(MODEL_PATH)
    except Exception:
        try:
            model = sm.load_results(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Model could not be loaded: {e}")

    try:
        features = pd.read_json(FEATURES_PATH, typ="series").to_list()
    except Exception as e:
        st.error(f"❌ Features file could not be loaded: {e}")

    return preprocessor, model, features


def premium_predictor() -> None:
    """
    Main function for the Premium Predictor page.
    """
    preprocessor, model, expected_features = load_premium_models()

    # Page Title
    st.title("💰 Prédicteur de Prime d'Assurance")
    st.markdown("Entrez vos détails pour obtenir une estimation de votre prime annuelle.")

    # User Input Form
    if not all([preprocessor, model, expected_features]):
        st.warning(
            "⚠️ Un ou plusieurs artefacts de modèle sont manquants."
            "Veuillez exécuter `uv run python src/models/final_model_premium.py`"
        )
        return

    with st.form(key="prediction_form_premium"):
        st.subheader("Détails du Véhicule")
        col1, col2 = st.columns(2)
        with col1:
            value_vehicle = st.number_input("Valeur du véhicule (€)", min_value=500, max_value=100000, value=20000, step=500)
            vehicle_age = st.slider("Âge du véhicule (années)", min_value=0, max_value=40, value=5)
            
            # --- CORRECTION 1 ---
            # Ajout de .get(x, str(x)) pour garantir un retour str
            type_risk = st.selectbox(
                "Type de véhicule",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "Moto", 2: "Camionnette", 3: "Voiture", 4: "Agricole"}.get(x, str(x))
            )
            # --- FIN CORRECTION 1 ---
            
        with col2:
            power = st.number_input("Puissance (ch)", min_value=40, max_value=500, value=110)
            
            # --- CORRECTION 2 ---
            # Ajout de .get(x, str(x)) pour garantir un retour str
            type_fuel = st.selectbox(
                "Carburant",
                options=[1, 2, 0],
                format_func=lambda x: {1: "Essence", 2: "Diesel", 0: "Autre"}.get(x, str(x))
            )
            # --- FIN CORRECTION 2 ---
            
            n_doors = st.selectbox("Nombre de portes", options=[0, 2, 3, 4, 5, 6],
                                   format_func=lambda x: f"{x} portes" if x > 0 else "N/A (ex: Moto)")

        st.subheader("Détails Conducteur & Contrat")
        col3, col4 = st.columns(2)
        with col3:
            driver_age = st.slider("Âge du conducteur", min_value=18, max_value=90, value=35)
            driving_experience = st.slider("Années d'expérience", min_value=0, max_value=72, value=15)
            area = st.radio("Zone", options=[0, 1],
                            format_func=lambda x: {0: "Rurale", 1: "Urbaine"}.get(x))
        with col4:
            seniority = st.slider("Ancienneté (années)", min_value=0, max_value=50, value=3)
            lapse = st.radio("Contrat déjà résilié ?", options=[0, 1],
                             format_func=lambda x: {0: "Non", 1: "Oui"}.get(x))
            second_driver = st.radio("Second conducteur ?", options=[0, 1],
                                     format_func=lambda x: {0: "Non", 1: "Oui"}.get(x))

        # Variables fixes requises par le preprocessor
        policies_in_force = 1
        payment = 0
        distribution_channel = 0

        submit_button = st.form_submit_button(label="Estimer ma Prime")

    # --- Prediction Logic ---
    if submit_button:
        # Créer le DataFrame d'une ligne pour le preprocessor
        input_data = pd.DataFrame({
            'seniority': [seniority],
            'policies_in_force': [policies_in_force],
            'max_policies': [policies_in_force],
            'max_products': [1],
            'lapse': [lapse],
            'power': [power],
            'cylinder_capacity': [1600],  # Dummy
            'value_vehicle': [value_vehicle],
            'length': [4.5],  # Dummy
            'weight': [1300],  # Dummy
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

        # Étape 1: Transformer les données
        input_processed = preprocessor.transform(input_data)

        # Étape 2: Recréer le DataFrame avec les bons noms de colonnes
        num_features = preprocessor.named_transformers_['num'].feature_names_in_
        cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
        all_feature_names = list(num_features) + list(cat_features)
        input_df = pd.DataFrame(input_processed, columns=all_feature_names)

        # Étape 3: Ajouter la constante
        input_df_with_const = sm.add_constant(input_df, prepend=True, has_constant="add")

        # Étape 4: Aligner les colonnes (CRUCIAL)
        input_aligned = pd.DataFrame(columns=expected_features, index=[0]).fillna(0.0)
        
        for col in input_aligned.columns:
            if col in input_df_with_const.columns:
                input_aligned[col] = input_df_with_const[col].values

        # Étape 5: Prédiction
        predicted_premium = model.predict(input_aligned)
        premium = float(predicted_premium[0])

        # Afficher le résultat
        st.success(f"## Prime Annuelle Estimée : €{premium:,.2f}")