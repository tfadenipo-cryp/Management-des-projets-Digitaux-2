import streamlit as st
import pandas as pd
import plotly.express as px  # type: ignore
from datetime import datetime


def bivariate_analysis(df: pd.DataFrame) -> None:
    """
    Perform interactive bivariate analyses on insurance premium data.
    Displays relationships between vehicle and driver attributes and the premium.
    """

    st.subheader("💰 Premium Analysis")
    st.markdown("""
    Explore how **vehicle value**, **age**, **driver demographics**, and **area type**
    influence insurance premiums.
    """)

    # ------------------------------------------------------------------
    # Ensure columns exist and are properly typed
    # ------------------------------------------------------------------
    for col in ["premium", "value_vehicle", "year_matriculation", "area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date_birth" in df.columns:
        df["date_birth"] = pd.to_datetime(df["date_birth"], errors="coerce")

    # Drop missing values in key columns
    df = df.dropna(subset=["premium"])

    # ------------------------------------------------------------------
    # Analysis type selection
    # ------------------------------------------------------------------
    analysis_type = st.selectbox(
        "Choose an analysis type:",
        [
            "Vehicle Value vs Premium",
            "Vehicle Age vs Premium",
            "Driver Age vs Premium",
            "Premium by Area",
        ],
    )

    # Define color palette
    colors = ["#0066cc", "#ffa600", "#00cc99", "#ff6361"]

    # ------------------------------------------------------------------
    # 1️⃣ Vehicle Value vs Premium
    # ------------------------------------------------------------------
    if analysis_type == "Vehicle Value vs Premium":
        if "value_vehicle" not in df.columns:
            st.warning("⚠️ Column 'value_vehicle' not found in dataset.")
            return

        clean_df = df.dropna(subset=["value_vehicle"])
        if clean_df.empty:
            st.warning("No valid data for vehicle value analysis.")
            return

        fig = px.scatter(
            clean_df,
            x="value_vehicle",
            y="premium",
            trendline="ols",
            color_discrete_sequence=[colors[0]],
            opacity=0.7,
            title="Vehicle Value vs Premium",
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color="DarkSlateGrey")))
        fig.update_layout(template="plotly_white", title_font_color=colors[0])
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 2️⃣ Vehicle Age vs Premium
    # ------------------------------------------------------------------
    elif analysis_type == "Vehicle Age vs Premium":
        if "year_matriculation" not in df.columns:
            st.warning("⚠️ Column 'year_matriculation' not found in dataset.")
            return

        clean_df = df.dropna(subset=["year_matriculation"])
        clean_df["vehicle_age"] = datetime.now().year - clean_df["year_matriculation"]

        fig = px.scatter(
            clean_df,
            x="vehicle_age",
            y="premium",
            trendline="ols",
            color_discrete_sequence=[colors[1]],
            opacity=0.6,
            title="Vehicle Age vs Premium",
        )
        fig.update_layout(template="plotly_white", title_font_color=colors[1])
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 3️⃣ Driver Age vs Premium
    # ------------------------------------------------------------------
    elif analysis_type == "Driver Age vs Premium":
        if "date_birth" not in df.columns:
            st.warning("⚠️ Column 'date_birth' not found in dataset.")
            return

        clean_df = df.dropna(subset=["date_birth"])
        clean_df["driver_age"] = (datetime.now() - clean_df["date_birth"]).dt.days / 365.25
        clean_df = clean_df[(clean_df["driver_age"] >= 18) & (clean_df["driver_age"] <= 90)]

        fig = px.scatter(
            clean_df,
            x="driver_age",
            y="premium",
            trendline="ols",
            color_discrete_sequence=[colors[2]],
            opacity=0.6,
            title="Driver Age vs Premium",
        )
        fig.update_layout(template="plotly_white", title_font_color=colors[2])
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 4️⃣ Premium by Area
    # ------------------------------------------------------------------
    elif analysis_type == "Premium by Area":
        if "area" not in df.columns:
            st.warning("⚠️ Column 'area' not found in dataset.")
            return

        clean_df = df.dropna(subset=["area"])
        clean_df["area_type"] = clean_df["area"].apply(lambda x: "Urban" if x == 1 else "Rural")

        fig = px.box(
            clean_df,
            x="area_type",
            y="premium",
            color="area_type",
            color_discrete_sequence=[colors[0], colors[1]],
            title="Premium by Geographic Area",
        )
        fig.update_layout(template="plotly_white", title_font_color=colors[3])
        st.plotly_chart(fig, use_container_width=True)

