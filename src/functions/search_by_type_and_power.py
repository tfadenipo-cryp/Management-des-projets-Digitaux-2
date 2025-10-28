import streamlit as st
import pandas as pd


def search_by_type_and_power(df: pd.DataFrame) -> None:
    """Displays the average claim cost by both vehicle type and power."""

    st.subheader("🔧 Claim Cost by Vehicle Type and Power")
    st.markdown("This section combines **vehicle type** and **power** to analyze claim cost variations.")

    required_cols = ["type_risk", "power", "cost_claims_year"]
    if not all(col in df.columns for col in required_cols):
        st.warning(f"⚠️ Missing required columns: {required_cols}")
        st.write("Available columns:", df.columns.tolist())
        return

    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle",
    }

    df["type_risk"] = pd.to_numeric(df["type_risk"], errors="coerce")
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df["cost_claims_year"] = pd.to_numeric(df["cost_claims_year"], errors="coerce")
    df = df.dropna(subset=["type_risk", "power", "cost_claims_year"])

    if df.empty:
        st.warning("⚠️ Not enough valid data for this analysis.")
        return

    df["vehicle_type"] = df["type_risk"].map(vehicle_type_map)

    selected_type = st.selectbox("Select a vehicle type:", sorted(df["vehicle_type"].dropna().unique()))

    # Ensure power list has integers only
    available_powers = sorted(df[df["vehicle_type"] == selected_type]["power"].dropna().unique())
    power_list = [int(p) for p in available_powers if not pd.isna(p)]

    selected_power = st.selectbox("Select a vehicle power (HP):", power_list)

    filtered_df = df[(df["vehicle_type"] == selected_type) & (df["power"] == selected_power)]

    if filtered_df.empty:
        st.warning("⚠️ No available records for this combination.")
        return

    avg_cost = filtered_df["cost_claims_year"].mean()

    st.markdown(
        f"""
        ### Results for {selected_type} ({selected_power} HP):
        - **Average yearly claim cost:** €{avg_cost:,.2f}
        - **Number of records:** {len(filtered_df)}
        """
    )
