import pandas as pd
import streamlit as st
from functions.utils.utils import display_results 
from functions.logic.analysis_type_logic import calculate_avg_cost_by_type # <-- Import Logique

# ======================================================================
# FEATURE 2: SEARCH BY VEHICLE 
# ======================================================================

# NOTE: La fonction calculate_avg_cost_by_type est maintenant dans analysis_type_logic.py

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

    avg_cost = calculate_avg_cost_by_type(df, selected_type) 

    if avg_cost is not None:
        display_results(f"Analysis for: {selected_type}", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("No data found for this vehicle type.")