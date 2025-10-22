import streamlit as st
import pandas as pd
import plotly.express as px  #type: ignore
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
    # Define a distinct trendline color
    trendline_color = colors[3] # This is the red color

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
        
        # --- FIX: Add outlier capping for readability ---
        prem_cap = clean_df["premium"].quantile(0.99)
        val_cap = clean_df["value_vehicle"].quantile(0.99)
        clean_df = clean_df[(clean_df["premium"] < prem_cap) & (clean_df["value_vehicle"] < val_cap)]

        fig = px.scatter(
            clean_df,
            x="value_vehicle",
            y="premium",
            trendline="ols",
            trendline_color_override=trendline_color,
            color_discrete_sequence=[colors[0]], # Markers will be blue
            opacity=0.7,
            title="Vehicle Value vs Premium",
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color="DarkSlateGrey")))
        fig.update_layout(template="plotly_white", title_font_color=colors[0])
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # ------------------------------------------------------------------
    # 2️⃣ Vehicle Age vs Premium
    # ------------------------------------------------------------------
    elif analysis_type == "Vehicle Age vs Premium":
        if "year_matriculation" not in df.columns:
            st.warning("⚠️ Column 'year_matriculation' not found in dataset.")
            return

        clean_df = df.dropna(subset=["year_matriculation"])
        if clean_df.empty:
            st.warning("No valid data for vehicle age analysis.")
            return
            
        clean_df["vehicle_age"] = datetime.now().year - clean_df["year_matriculation"]
        
        # --- FIX: Add outlier capping for readability ---
        prem_cap = clean_df["premium"].quantile(0.99)
        clean_df = clean_df[clean_df["premium"] < prem_cap]

        fig = px.scatter(
            clean_df,
            x="vehicle_age",
            y="premium",
            trendline="ols",
            trendline_color_override=trendline_color,
            color_discrete_sequence=[colors[1]], # Markers will be orange
            opacity=0.6,
            title="Vehicle Age vs Premium",
        )
        fig.update_layout(template="plotly_white", title_font_color=colors[1])
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # ------------------------------------------------------------------
    # 3️⃣ Driver Age vs Premium
    # ------------------------------------------------------------------
    elif analysis_type == "Driver Age vs Premium":
        if "date_birth" not in df.columns:
            st.warning("⚠️ Column 'date_birth' not found in dataset.")
            return

        clean_df = df.dropna(subset=["date_birth"])
        if clean_df.empty:
            st.warning("No valid data for driver age analysis.")
            return
            
        clean_df["driver_age"] = (datetime.now() - clean_df["date_birth"]).dt.days / 365.25
        # --- FIX: Corrected typo from "driver_.age" to "driver_age" ---
        clean_df = clean_df[(clean_df["driver_age"] >= 18) & (clean_df["driver_age"] <= 90)]
        
        # --- FIX: Add outlier capping for readability ---
        prem_cap = clean_df["premium"].quantile(0.99)
        clean_df = clean_df[clean_df["premium"] < prem_cap]

        fig = px.scatter(
            clean_df,
            x="driver_age",
            y="premium",
            trendline="ols",
            trendline_color_override=trendline_color,
            color_discrete_sequence=[colors[2]], # Markers will be green
            opacity=0.6,
            title="Driver Age vs Premium",
        )
        fig.update_layout(template="plotly_white", title_font_color=colors[2])
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # ------------------------------------------------------------------
    # 4️⃣ Premium by Area
    # ------------------------------------------------------------------
    elif analysis_type == "Premium by Area":
        if "area" not in df.columns:
            st.warning("⚠️ Column 'area' not found in dataset.")
            return

        clean_df = df.dropna(subset=["area"])
        if clean_df.empty:
            st.warning("No valid data for area analysis.")
            return
            
        clean_df["area_type"] = clean_df["area"].apply(lambda x: "Urban" if x == 1 else "Rural")
        
        # --- FIX: Add outlier capping for readability ---
        prem_cap = clean_df["premium"].quantile(0.99)
        clean_df = clean_df[clean_df["premium"] < prem_cap]

        fig = px.box(
            clean_df,
            x="area_type",
            y="premium",
            color="area_type",
            color_discrete_sequence=[colors[0], colors[1]], # Box plots will be blue and orange
            title="Premium by Geographic Area",
        )
        fig.update_layout(template="plotly_white", title_font_color=colors[3])
        st.plotly_chart(fig, use_container_width=True, theme=None)

