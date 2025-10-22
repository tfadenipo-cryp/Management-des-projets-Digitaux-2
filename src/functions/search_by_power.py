import streamlit as st
import pandas as pd


def search_by_power(df: pd.DataFrame) -> None:
    st.subheader("🔍 Search by Vehicle Power")

    if "power" not in df.columns or "cost_claims_year" not in df.columns:
        st.warning("⚠️ Missing required columns: 'power' or 'cost_claims_year'.")
        return

    power_list = sorted(df["power"].dropna().unique())
    power_choice = st.selectbox("Select vehicle horsepower:", power_list)

    avg_data = df.groupby("power")[["cost_claims_year"]].mean().reset_index()
    row = avg_data[avg_data["power"] == power_choice]

    if not row.empty:
        avg_cost = row["cost_claims_year"].values[0]
        st.write(f"**Average Annual Claim Cost:** ${avg_cost:,.2f}")
    else:
        st.warning("No data found for this power value.")
