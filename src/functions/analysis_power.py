import pandas as pd
import streamlit as st
from functions.utils import display_results 

# ======================================================================
# FEATURE 1: SEARCH BY VEHICLE POWER
# ======================================================================

def search_by_power(df: pd.DataFrame) -> None:
    """Renders interface and displays average claim cost by vehicle power."""
    
    power_list: list[float] = sorted(df["power"].dropna().unique().tolist())
    power_choice = st.selectbox("Select the vehicle's horsepower:", power_list)

    if power_choice is None:
        st.warning("Please select a valid power value.")
        return

    df_clean = df.dropna(subset=["power", "cost_claims_year"])
    avg_data = df_clean.groupby("power", as_index=False)["cost_claims_year"].mean()
    row = avg_data[avg_data["power"] == power_choice]

    if not row.empty:
        avg_cost: float = float(row["cost_claims_year"].values[0])
        display_results(f"Analysis for: {int(power_choice)} HP", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("No records found for this vehicle power.")