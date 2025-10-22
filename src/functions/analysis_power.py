import pandas as pd
import streamlit as st
from functions.utils import display_results 

# ======================================================================
# FEATURE 1: SEARCH BY VEHICLE POWER 
# ======================================================================

def calculate_avg_cost_by_power(df: pd.DataFrame, power_choice: float) -> float | None:
    """Calculates the average claim cost for a specific vehicle power (TESTABLE LOGIC)."""
    
    df_clean = df.dropna(subset=["power", "cost_claims_year"])
    
    avg_data = df_clean.groupby("power", as_index=False)["cost_claims_year"].mean()
    row = avg_data[avg_data["power"] == power_choice]

    if not row.empty:
        return float(row["cost_claims_year"].values[0])
    
    return None

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