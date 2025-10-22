import pandas as pd
import streamlit as st
from functions.utils.utils import display_results 
from functions.logic.analysis_power_logic import calculate_avg_cost_by_power # <-- Import Logique

# ======================================================================
# FEATURE 1: SEARCH BY VEHICLE POWER 
# ======================================================================

# NOTE: La fonction calculate_avg_cost_by_power est maintenant dans analysis_power_logic.py

def search_by_power(df: pd.DataFrame) -> None:
    """Renders interface and displays average claim cost by vehicle power."""
    
    power_list: list[float] = sorted(df["power"].dropna().unique().tolist())
    power_choice = st.selectbox("Select the vehicle's horsepower:", power_list)

    if power_choice is None:
        st.warning("Please select a valid power value.")
        return

    avg_cost = calculate_avg_cost_by_power(df, power_choice) 

    if avg_cost is not None:
        display_results(f"Analysis for: {int(power_choice)} HP", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("No records found for this vehicle power.")