import streamlit as st
import pandas as pd


def search_by_vehicle_type(df: pd.DataFrame) -> None:
    """Displays the average claim cost by vehicle type."""

    st.subheader("Claim Cost by Vehicle Type")
    st.markdown(
        "This section shows the **average annual claim cost** depending on the vehicle category."
    )

    required_cols = ["type_risk", "cost_claims_year"]
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Missing required columns: {required_cols}")
        st.write("Available columns:", df.columns.tolist())
        return

    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle",
    }

    df["type_risk"] = pd.to_numeric(df["type_risk"], errors="coerce")
    df["cost_claims_year"] = pd.to_numeric(df["cost_claims_year"], errors="coerce")
    df = df.dropna(subset=["type_risk", "cost_claims_year"])

    if df.empty:
        st.warning("Not enough valid data for this analysis.")
        return

    df["vehicle_type"] = df["type_risk"].map(vehicle_type_map)
    available_types = df["vehicle_type"].dropna().unique().tolist()

    selected_type = st.selectbox("Select a vehicle type:", available_types)
    filtered_df = df[df["vehicle_type"] == selected_type]
    avg_cost = filtered_df["cost_claims_year"].mean()

    st.markdown(
        f"""
        ### Results for {selected_type}:
        - **Average yearly claim cost:** €{avg_cost:,.2f}
        - **Number of records:** {len(filtered_df)}
        """
    )
