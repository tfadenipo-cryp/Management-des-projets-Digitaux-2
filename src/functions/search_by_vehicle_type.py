import streamlit as st
import pandas as pd


def search_by_vehicle_type(df: pd.DataFrame) -> None:
    st.subheader("🚗 Search by Vehicle Type")

    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle"
    }

    if "type_risk" not in df.columns or "cost_claims_year" not in df.columns:
        st.warning("⚠️ Missing required columns: 'type_risk' or 'cost_claims_year'.")
        return

    df["vehicle_type"] = df["type_risk"].map(vehicle_type_map)
    selected_type = st.selectbox("Select vehicle type:", df["vehicle_type"].dropna().unique())

    avg_cost = df[df["vehicle_type"] == selected_type]["cost_claims_year"].mean()
    if not pd.isna(avg_cost):
        st.write(f"**Average Annual Claim Cost:** ${avg_cost:,.2f}")
    else:
        st.warning("No data found for this type.")

