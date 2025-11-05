import sys
from pathlib import Path
import textwrap
import streamlit as st

# ------------------------------------------------------------------------------
# Project root → ensure imports work
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from functions.load_data import load_data
from functions.search_by_power import search_by_power
from functions.search_by_vehicle_type import search_by_vehicle_type
from functions.search_by_type_and_power import search_by_type_and_power
from functions.variable_analysis import variable_analysis
from functions.bivariate_analysis import bivariate_analysis
from functions.premium_predictor import premium_predictor


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main() -> None:

    st.set_page_config(page_title="Vehicle Insurance Dashboard", layout="wide")
    st.title("Vehicle Insurance Data Dashboard")

    st.header("Project & Dataset Overview")
    st.markdown(
        textwrap.dedent("""
        <p style="text-align: justify;">
        This interactive platform is developed as part of the <b>Management of Digital Projects 2 (MPD2)</b> course. 
        It provides a modular environment for exploring and analyzing a comprehensive 
        <b>Motor Vehicle Insurance dataset</b>.
        </p>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---------------------------------------------
    # Data
    # ---------------------------------------------
    df = load_data()
    if df is None or df.empty:
        st.error("⚠️ Unable to load dataset.")
        st.stop()

    st.caption(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    st.divider()

    # ---------------------------------------------
    # Navigation state
    # ---------------------------------------------
    if "page" not in st.session_state:
        st.session_state.page = "home"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔍 Risk Analysis"):
            st.session_state.page = "risk"
    with col2:
        if st.button("📊 Variable Exploration"):
            st.session_state.page = "exploration"
    with col3:
        if st.button("💰 Premium Analysis"):
            st.session_state.page = "premium"
    with col4:
        if st.button("🔮 Premium Predictor"):
            st.session_state.page = "predictor"

    # ---------------------------------------------
    # Page router
    # ---------------------------------------------
    if st.session_state.page == "risk":
        menu = st.selectbox(
            "Choose analysis:",
            ["Search by Vehicle Power", "Search by Vehicle Type", "Search by Vehicle Type AND Power"],
        )

        if menu == "Search by Vehicle Power":
            search_by_power(df)
        elif menu == "Search by Vehicle Type":
            search_by_vehicle_type(df)
        else:
            search_by_type_and_power(df)

        if st.button("⬅️ Back"):
            st.session_state.page = "home"
            st.rerun()

    elif st.session_state.page == "exploration":
        variable_analysis(df)

        if st.button("⬅️ Back"):
            st.session_state.page = "home"
            st.rerun()

    elif st.session_state.page == "premium":
        bivariate_analysis(df)

        if st.button("⬅️ Back"):
            st.session_state.page = "home"
            st.rerun()

    elif st.session_state.page == "predictor":
        premium_predictor()

        if st.button("⬅️ Back"):
            st.session_state.page = "home"
            st.rerun()


if __name__ == "__main__":
    main()
