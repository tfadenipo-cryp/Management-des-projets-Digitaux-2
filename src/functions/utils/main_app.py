"""
Core application logic: Configuration, Data Loading, and Page Dispatcher.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import textwrap

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions.engineering import get_processed_data
from functions.visualization import explore_variables
from functions.analysis_power import search_by_power
from functions.analysis_type import search_by_vehicle_type
from functions.bivariate import render_bivariate_analysis 
from functions.analysis_combined import search_by_type_and_power # <-- NOUVEL IMPORT

# ======================================================================
# DATA LOADING & CONFIGURATION
# ======================================================================

@st.cache_data
def load_data_pipeline():
    """Executes the data loading and preprocessing pipeline."""
    with st.spinner("Preparing and cleaning the database..."):
        df = get_processed_data()
    return df

# Ajout de la nouvelle page au map
PAGE_MAP = {
    "HOME": "Home",
    "RISK_ANALYSIS": "Specific Risk Analysis",
    "VARIABLE_EXPLORATION": "Variable Exploration",
    "BIVARIATE_ANALYSIS": "Bivariate Analysis",
    "COMBINED_RISK": "Combined Risk Analysis" # Nouvelle page (si vous vouliez un nouveau bouton principal)
}


# ======================================================================
# NAVIGATION UTILS & CSS (Le code CSS reste inchangé)
# ======================================================================

def inject_css():
    """Injects custom CSS to hide Streamlit header/footer and ensure text alignment."""
    st.markdown("""
        <style>
            /* Global text justification (for Streamlit markdown blocks) */
            p { text-align: justify; }
            .st-markdown div { text-align: justify; }
            
            /* Hides the Streamlit menu and footer for a cleaner look */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            /* Style for the portal buttons */
            .stButton>button {
                width: 100%;
                border-radius: 8px;
            }
        </style>
    """, unsafe_allow_html=True)


def go_home():
    """Sets state and reruns to navigate back to the Home page."""
    st.session_state.page = "HOME"
    st.rerun()

def render_back_home_button():
    """Renders a button to return to the Home page."""
    st.markdown("---")
    if st.button("⬅️ Back to Home", use_container_width=False):
        go_home()


# ======================================================================
# PAGE RENDERING FUNCTIONS (Mise à jour pour le nouvel onglet)
# ======================================================================

def render_home_page(df: pd.DataFrame):
    """
    Renders the main Home portal page with simplified navigation and no metrics.
    """
    st.title("Car Insurance Data Dashboard")
    st.markdown("---")

    st.header("Project & Dataset Overview")
    presentation_text = textwrap.dedent("""
        <p style="text-align: justify;">
        This interactive platform is developed as part of the <b>Management of Digital Projects 2 (MPD2)</b> course. 
        It provides a modular and robust environment for exploring and analyzing a comprehensive <b>Motor Vehicle Insurance dataset</b>. 
        The project adheres to principles of clean coding and modular architecture to ensure scalability and independent unit testing.
        </p>
    """)
    st.markdown(presentation_text, unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("Feature Navigation Portal")
    st.markdown('<p style="text-align:center;">Select a functional area below to start your analysis.</p>', unsafe_allow_html=True)

    cols = st.columns(3)

    with cols[0]:
        st.subheader("Specific Risk Analysis")
        st.markdown('<p>Analysis by Vehicle Power and Risk Type (includes combined search).</p>', unsafe_allow_html=True)
        if st.button("Start Risk Analysis", key="btn_analysis"):
            st.session_state.page = "RISK_ANALYSIS"
            st.rerun()

    with cols[1]:
        st.subheader("Variable Exploration")
        st.markdown('<p>Global visualization of variable distributions.</p>', unsafe_allow_html=True)
        if st.button("Start Exploration", key="btn_exploration"):
            st.session_state.page = "VARIABLE_EXPLORATION"
            st.rerun()

    with cols[2]:
        st.subheader("Bivariate Analysis")
        st.markdown('<p>Analysis of premium drivers (Vehicle Value, Age, Driver Age, Area).</p>', unsafe_allow_html=True)
        if st.button("Start Bivariate Analysis", key="btn_bivariate"):
            st.session_state.page = "BIVARIATE_ANALYSIS"
            st.rerun()


def render_analysis_page(df: pd.DataFrame):
    """Renders the page for specific analyses from colleagues 1 & 2."""
    st.title("Specific Risk and Claims Analysis")
    st.markdown("---")

    # AJOUT DU NOUVEL ONGLETTI
    tab_power, tab_type, tab_combined = st.tabs([
        "Analysis by Vehicle Power", 
        "Analysis by Risk Type",
        "Combined Search" # <-- NOUVEL ONGLETTI
    ])
    
    with tab_power:
        st.header("Analysis by Vehicle Power")
        search_by_power(df)

    with tab_type:
        st.header("Analysis by Risk Type")
        search_by_vehicle_type(df)
        
    with tab_combined: # <-- Rendu de la nouvelle fonction ici
        search_by_type_and_power(df)
        
    render_back_home_button()


def render_visualization_page(df: pd.DataFrame):
    """Renders the generic variable exploration page."""
    st.title("Data Visualization & Exploration")
    st.markdown("---")
    explore_variables(df) 
    render_back_home_button()


def render_bivariate_analysis_page(df: pd.DataFrame):
    """Renders the dedicated page for the Bivariate Analysis."""
    st.title("Bivariate Analysis: Premium Drivers")
    st.markdown("---")
    render_bivariate_analysis(df) 
    render_back_home_button()


def run_dashboard():
    """
    Main function to run the Streamlit application.
    Handles configuration, data loading, and page dispatch based on session state.
    """
    inject_css()
    
    st.set_page_config(
        page_title="Car Insurance Data Analysis",
        layout="wide",
        initial_sidebar_state="collapsed", 
    )
    
    if 'page' not in st.session_state:
        st.session_state.page = "HOME"

    data_df = load_data_pipeline()

    if data_df.empty:
        st.error("Application setup failed: Data loading error.")
        return

    if st.session_state.page == "HOME":
        render_home_page(data_df)
    elif st.session_state.page == "RISK_ANALYSIS":
        render_analysis_page(data_df)
    elif st.session_state.page == "VARIABLE_EXPLORATION":
        render_visualization_page(data_df)
    elif st.session_state.page == "BIVARIATE_ANALYSIS":
        render_bivariate_analysis_page(data_df)