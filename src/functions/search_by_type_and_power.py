import streamlit as st
import pandas as pd


def search_by_type_and_power(df: pd.DataFrame) -> None:
    st.subheader("⚙️ Combined Search by Vehicle Type and Power")

    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle"
    }

    df["vehicle_type"] = df["type_risk"].map(vehicle_type_map)

    selected_type = st.selectbox("Select vehicle type:", df["vehicle_type"].dropna().unique())
    available_powers = sorted(df[df["vehicle_type"] == selected_type]["power"].dropna().unique())
    selected_power = st.selectbox("Select horsepower:", available_powers)

    filtered = df[(df["vehicle_type"] == selected_type) & (df["power"] == selected_power)]

    if not filtered.empty:
        avg = filtered["cost_claims_year"].mean()
        st.write(f"**Average Annual Claim Cost:** ${avg:,.2f}")
    else:
        st.warning("No records found for this combination.")
