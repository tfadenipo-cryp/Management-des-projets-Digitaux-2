#Fonction de mise en place du streamlitt


import sys
from pathlib import Path
import streamlit as st
import textwrap

# Dynamically add src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import (# noqa: E402
    load_data,
    search_by_power,
    search_by_vehicle_type,
    search_by_type_and_power,
    variable_analysis,
    bivariate_analysis,
)


def main():
    st.set_page_config(page_title="Vehicle Insurance Dashboard", layout="wide")
    st.title(" Vehicle Insurance Data Dashboard")
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

    df = load_data()
    if df.empty:
        st.error("Error loading dataset.")
        return

    st.divider()
    st.subheader("Navigation")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Risk Analysis"):
            st.session_state.page = "risk"
    with col2:
        if st.button("📊 Variable Exploration"):
            st.session_state.page = "exploration"
    with col3:
        if st.button("💰 Premium Analysis"):
            st.session_state.page = "premium"

    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "risk":
        menu = st.selectbox("Choose analysis:", [
            "Search by Vehicle Power", "Search by Vehicle Type", "Search by Vehicle Type AND Power"
        ])
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


if __name__ == "__main__":
    main()

