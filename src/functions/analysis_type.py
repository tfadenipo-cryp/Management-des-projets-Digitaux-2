import pandas as pd
import streamlit as st
from functions.utils import display_results 

# ======================================================================
# FEATURE 2: SEARCH BY VEHICLE TYPE
# ======================================================================

def search_by_vehicle_type(df: pd.DataFrame) -> None:
    """Renders interface and displays average claim cost by risk type."""
    
    vehicle_type_map: dict[int, str] = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle",
    }
    inverse_map: dict[str, int] = {v: k for k, v in vehicle_type_map.items()}

    vehicle_types = list(inverse_map.keys())
    selected_type = st.selectbox("Select a vehicle type:", vehicle_types)

    if not selected_type:
        st.warning("Please select a valid vehicle type.")
        return

    df_clean = df.dropna(subset=["type_risk", "cost_claims_year"]).copy()
    df_clean["Vehicle_Type_Label"] = df_clean["type_risk"].map(vehicle_type_map)
    avg_data = df_clean.groupby("Vehicle_Type_Label", as_index=False)["cost_claims_year"].mean()
    row = avg_data[avg_data["Vehicle_Type_Label"] == selected_type]

    if not row.empty:
        avg_cost = float(row["cost_claims_year"].values[0])
        display_results(f"Analysis for: {selected_type}", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("No data found for this vehicle type.")