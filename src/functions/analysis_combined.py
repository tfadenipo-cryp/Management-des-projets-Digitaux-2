import streamlit as st
import pandas as pd
from functions.utils import display_results
from functions.logic.analysis_combined_logic import calculate_avg_cost_combined

# ======================================================================
# FEATURE 4: COMBINED SEARCH (Interface)
# ======================================================================

def search_by_type_and_power(df: pd.DataFrame) -> None:
    """Renders interface and displays average claim cost by vehicle type AND power."""
    st.subheader("⚙️ Combined Search by Vehicle Type and Power")

    # Prétraitement léger nécessaire à l'interface Streamlit (pour les selectbox)
    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle"
    }
    df_temp = df.dropna(subset=["type_risk", "power"]).copy()
    df_temp["vehicle_type"] = df_temp["type_risk"].map(vehicle_type_map)

    # 1. Sélection du type
    selected_type = st.selectbox("Select vehicle type:", df_temp["vehicle_type"].dropna().unique())
    
    # 2. Sélection de la puissance (filtrée par le type)
    available_powers = sorted(df_temp[df_temp["vehicle_type"] == selected_type]["power"].dropna().unique())
    selected_power = st.selectbox("Select horsepower:", available_powers)

    if selected_type is None or selected_power is None:
        return # Attendre une sélection valide

    # 3. Appel de la logique
    avg_cost = calculate_avg_cost_combined(df, selected_type, selected_power)

    # 4. Affichage des résultats
    if avg_cost is not None:
        display_results(f"Combined Analysis: {selected_type} @ {int(selected_power)} HP", {
            "Average Annual Claim Cost": f"${avg_cost:,.2f}"
        })
    else:
        st.warning("No records found for this combination.")