"""
Streamlit Page: Probability Predictor (for Insurer/Décideur)
(Nom de fichier conservé : cost_predictor.py pour la compatibilité des imports)
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib  # type: ignore
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
import streamlit as st

# --- Project paths ---
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[2]
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"

# --- Artifact Stems ---
# Nous chargeons les fichiers "cost" qui contiennent maintenant
# le modèle de PROBABILITÉ
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

    # --- CORRECTION : Titre mis à jour ---
    st.title("⚖️ Prédicteur de Probabilité de Sinistre")
    st.markdown(
        "Estimez la **probabilité qu'un profil ait un sinistre** au cours de l'année."
    )

    if not all([preprocessor, model, expected_features]):
        st.warning(
            "⚠️ Un ou plusieurs artefacts de modèle sont manquants."
            "Veuillez exécuter `uv run python src/models/final_model_cost.py`"
        )
        return

    # --- Prediction Form ---
    with st.form("prediction_form_cost"):
        st.subheader("Véhicule")
        col1, col2 = st.columns(2)
        with col1:
            value_vehicle = st.number_input("Valeur du véhicule (€)", 500, 100000, 20000)
            vehicle_age = st.slider("Âge du véhicule (années)", 0, 40, 5)
            type_risk = st.selectbox(
                "Type de véhicule",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Moto", 2: "Camionnette", 3: "Voiture", 4: "Agricole"
                }[x],
            )
        with col2:
            power = st.number_input("Puissance (ch)", 40, 500, 110)
            type_fuel = st.selectbox(
                "Carburant",
                [1, 2, 0],
                format_func=lambda x: {1: "Essence", 2: "Diesel", 0: "Autre"}[x],
            )
            n_doors = st.selectbox("Nombre de portes", [0, 2, 3, 4, 5, 6])

        st.subheader("Conducteur / Contrat")
        col3, col4 = st.columns(2)
        with col3:
            driver_age = st.slider("Âge du conducteur", 18, 90, 35)
            driving_experience = st.slider("Années d'expérience", 0, 72, 15)
            area = st.radio("Zone", [0, 1], format_func=lambda x: {0: "Rurale", 1: "Urbaine"}[x])
        with col4:
            seniority = st.slider("Ancienneté (années)", 0, 50, 3)
            lapse = st.radio("Contrat déjà résilié ?", [0, 1], format_func=lambda x: {0: "Non", 1: "Oui"}[x])
            second_driver = st.radio("Second conducteur ?", [0, 1], format_func=lambda x: {0: "Non", 1: "Oui"}[x])

        submit = st.form_submit_button("Estimer la Probabilité de Risque")

    if not submit:
        return

    # --- Build Input DataFrame ---
    df = pd.DataFrame({
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
        "policies_in_force": [1],
        "max_policies": [1],
        "max_products": [1],
        "cylinder_capacity": [1600],
        "length": [4.5],
        "weight": [1300],
        "distribution_channel": [0],
        "payment": [0],
    })

    # --- Preprocess ---
    X_processed = preprocessor.transform(df)

    num_features = preprocessor.named_transformers_["num"].feature_names_in_
    cat_features = (
        preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out()
    )
    all_processed_features = list(num_features) + list(cat_features)

    X_df = pd.DataFrame(X_processed, columns=all_processed_features)
    X_df_sm = sm.add_constant(X_df, prepend=True, has_constant="add")

    X_final = pd.DataFrame(columns=expected_features, index=[0]).fillna(0.0)
    
    for col in X_final.columns:
        if col in X_df_sm.columns:
            X_final[col] = X_df_sm[col].values

    # --- Predict ---
    try:
        probability = model.predict(X_final)
        prob_percent = probability[0] * 100

        st.success(f"## ✅ Probabilité de sinistre : {prob_percent:,.1f} %")
        
        if prob_percent < 15:
            st.info("Risque faible")
        elif prob_percent < 30:
            st.warning("Risque modéré")
        else:
            st.error("Risque élevé")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
