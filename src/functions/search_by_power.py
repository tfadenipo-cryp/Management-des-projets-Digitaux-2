import streamlit as st
import pandas as pd


def search_by_power(df: pd.DataFrame) -> None:
    """Displays the average claim cost by vehicle power."""

    st.subheader("Claim Cost by Vehicle Power")
    st.markdown("This section displays the **average annual claim cost** depending on the vehicle's horsepower.")

    required_cols = ["power", "cost_claims_year"]
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Missing required columns: {required_cols}")
        st.write("Available columns:", df.columns.tolist())
        return

    # Type conversion
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df["cost_claims_year"] = pd.to_numeric(df["cost_claims_year"], errors="coerce")
    df = df.dropna(subset=["power", "cost_claims_year"])

    if df.empty:
        st.warning("Not enough valid data for this analysis.")
        return

    # Dropdown for power selection (convert to int to remove decimals)
    power_list = sorted(df["power"].dropna().unique())
    power_list = [int(p) for p in power_list if not pd.isna(p)]

    selected_power = st.selectbox("Select a vehicle power (HP):", power_list)

    # Filter and compute mean
    filtered_df = df[df["power"] == selected_power]
    avg_cost = filtered_df["cost_claims_year"].mean()

    st.markdown(
        f"""
        ### Results for {selected_power} HP:
        - **Average yearly claim cost:** €{avg_cost:,.2f}
        - **Number of records:** {len(filtered_df)}
        """
    )

