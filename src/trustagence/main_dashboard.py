"""
Main Dashboard Router
Handles navigation between Client and Insurer sections.
"""

# from __future__ import annotations

import streamlit as st
import textwrap
import pandas as pd  # Necessary to pass df to src.trustagence.analysis

from trustagence.analysis.search_by_power import search_by_power
from trustagence.engineering.load_data import load_data
from trustagence.analysis.search_by_vehicle_type import search_by_vehicle_type
from trustagence.analysis.search_by_type_and_power import search_by_type_and_power
from trustagence.analysis.variable_analysis import variable_analysis
from trustagence.analysis.bivariate_analysis import bivariate_analysis
from trustagence.model.premium_predictor import premium_predictor
from trustagence.model.cost_predictor import cost_predictor  # The filename is the same


# Import all page src.trustagence.analysis


# functions


def show_home_page() -> None:
    """
    Displays the main Home page with persona selection.
    """
    st.header("Welcome to the Auto Insurance Dashboard")
    st.markdown(
        textwrap.dedent("""
        <p style="text-align: justify;">
        This interactive platform is developed as part of the <b>Digital Project Management 2 (MPD2)</b> course.
        It provides an environment to explore and analyze a dataset
        on motor vehicle insurance.
        </p>
        <p>
        Please select your profile to access the tools dedicated to you.
        </p>
        """),
        unsafe_allow_html=True,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Client Area")
        st.markdown("Estimate your insurance premium and explore public data.")
        if st.button("Go to Client Area"):
            st.session_state.page = "client"
            st.rerun()

    with col2:
        st.subheader("Insurer Area")
        st.markdown("Access risk analysis and cost prediction tools.")
        if st.button("Go to Insurer Area"):
            # Use 'insurer' for session state consistency
            st.session_state.page = "insurer"
            st.rerun()


def show_client_page(df: pd.DataFrame) -> None:
    """
    Displays the 'Client' dashboard with all existing analyses.
    """
    if st.button("⬅️ Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("Client Area")

    menu = st.selectbox(
        "Choose an analysis:",
        [
            "Premium Predictor",
            "Premium Analysis (Bivariate)",
            "Variable Exploration",
            "Risk Analysis (by Power)",
            "Risk Analysis (by Type)",
            "Risk Analysis (by Type and Power)",
        ],
    )

    st.divider()

    # Router for Client page
    if menu == "Premium Predictor":
        premium_predictor()
    elif menu == "Premium Analysis (Bivariate)":
        bivariate_analysis(df)
    elif menu == "Variable Exploration":
        variable_analysis(df)
    elif menu == "Risk Analysis (by Power)":
        search_by_power(df)
    elif menu == "Risk Analysis (by Type)":
        search_by_vehicle_type(df)
    elif menu == "Risk Analysis (by Type and Power)":
        search_by_type_and_power(df)


def show_insurer_page(df: pd.DataFrame) -> None:
    """
    Displays the 'Insurer' dashboard.
    (Previously 'Décideur')
    """
    if st.button("⬅️ Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("Insurer Area")

    # --- CORRECTION: Menu text updated ---
    menu = st.selectbox(
        "Choose an analysis:",
        [
            "Risk Predictor (Probability)",
            # "Other analysis (coming soon)..."
        ],
    )

    st.divider()

    if menu == "Risk Predictor (Probability)":
        cost_predictor()  # The function is still called cost_predictor
    # elif menu == "Other analysis (coming soon)...":
    #    st.info("Coming soon.")


def run_dashboard() -> None:
    """Main Streamlit app router."""

    st.set_page_config(page_title="Insurance Dashboard", layout="wide")

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # --- Data Loading (once) ---
    df = load_data()
    if df is None or df.empty:
        st.error("⚠️ Could not load the dataset.")
        st.stop()

    # --- Page Router ---
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "client":
        show_client_page(df)
    elif st.session_state.page == "insurer":  # Changed from 'decideur'
        show_insurer_page(df)  # Call the renamed function
    else:
        # Default fallback to home
        st.session_state.page = "home"
        st.rerun()
