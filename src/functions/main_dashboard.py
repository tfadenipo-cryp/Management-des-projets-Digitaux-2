"""
Main Dashboard Router
Handles navigation between Client and Insurer (Décideur) sections.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import textwrap
import pandas as pd  # Nécessaire pour passer df aux fonctions

# --- Project root → ensure imports work ---
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
# --- End of sys.path modification ---

# Import all page functions
try:
    from functions.load_data import load_data
    from functions.search_by_power import search_by_power
    from functions.search_by_vehicle_type import search_by_vehicle_type
    from functions.search_by_type_and_power import search_by_type_and_power
    from functions.variable_analysis import variable_analysis
    from functions.bivariate_analysis import bivariate_analysis
    from functions.premium_predictor import premium_predictor
    from functions.cost_predictor import cost_predictor  # Le nom de fichier est le même
except ImportError as e:
    st.error(f"Erreur d'importation : {e}")
    st.stop()


def show_home_page() -> None:
    """
    Displays the main Home page with persona selection.
    """
    st.header("Bienvenue sur le Dashboard d'Assurance Auto")
    st.markdown(
        textwrap.dedent("""
        <p style="text-align: justify;">
        Cette plateforme interactive est développée dans le cadre du cours de <b>Management des Projets Digitaux 2 (MPD2)</b>. 
        Elle fournit un environnement pour explorer et analyser un jeu de données 
        sur l'assurance de véhicules à moteur.
        </p>
        <p>
        Veuillez sélectionner votre profil pour accéder aux outils qui vous sont dédiés.
        </p>
        """),
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Espace Client")
        st.markdown("Estimez votre prime d'assurance et explorez les données publiques.")
        if st.button("Accéder à l'Espace Client"):
            st.session_state.page = "client"
            st.rerun()
            
    with col2:
        st.subheader("👔 Espace Décideur")
        st.markdown("Accédez aux outils d'analyse de risque et de prédiction des coûts.")
        if st.button("Accéder à l'Espace Décideur"):
            st.session_state.page = "decideur"
            st.rerun()



def show_client_page(df: pd.DataFrame) -> None:
    """
    Displays the 'Client' dashboard with all existing analyses.
    """
    if st.button("⬅️ Accueil"):
        st.session_state.page = "home"
        st.rerun()
        
    st.title("👤 Espace Client")
    
    menu = st.selectbox(
        "Choisissez une analyse :",
        [
            "🔮 Prédicteur de Prime",
            "💰 Analyse de la Prime (Bivariée)",
            "📊 Exploration des Variables",
            "⚙️ Analyse Risque (par Puissance)",
            "🚘 Analyse Risque (par Type)",
            "🔧 Analyse Risque (par Type et Puissance)",
        ],
    )
    
    st.divider()

    # Router for Client page
    if menu == "🔮 Prédicteur de Prime":
        premium_predictor()
    elif menu == "💰 Analyse de la Prime (Bivariée)":
        bivariate_analysis(df)
    elif menu == "📊 Exploration des Variables":
        variable_analysis(df)
    elif menu == "⚙️ Analyse Risque (par Puissance)":
        search_by_power(df)
    elif menu == "🚘 Analyse Risque (par Type)":
        search_by_vehicle_type(df)
    elif menu == "🔧 Analyse Risque (par Type et Puissance)":
        search_by_type_and_power(df)


def show_decideur_page(df: pd.DataFrame) -> None:
    """
    Displays the 'Décideur' dashboard.
    """
    if st.button("⬅️ Accueil"):
        st.session_state.page = "home"
        st.rerun()
        
    st.title("👔 Espace Décideur")

    # --- CORRECTION : Texte du menu mis à jour ---
    menu = st.selectbox(
        "Choisissez une analyse :",
        [
            "⚖️ Prédicteur de Risque (Probabilité)",
            # "Autre analyse (à venir)..."
        ],
    )
    
    st.divider()
    
    if menu == "⚖️ Prédicteur de Risque (Probabilité)":
        cost_predictor() # La fonction s'appelle toujours cost_predictor
    # elif menu == "Autre analyse (à venir)...":
    #    st.info("Bientôt disponible.")


def main() -> None:
    """Main Streamlit app router."""

    st.set_page_config(page_title="Dashboard Assurance", layout="wide")
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # --- Data Loading (once) ---
    df = load_data()
    if df is None or df.empty:
        st.error("⚠️ Impossible de charger le jeu de données.")
        st.stop()
    
    # --- Page Router ---
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "client":
        show_client_page(df)
    elif st.session_state.page == "decideur":
        show_decideur_page(df)
    else:
        st.session_state.page = "home"
        st.rerun()
