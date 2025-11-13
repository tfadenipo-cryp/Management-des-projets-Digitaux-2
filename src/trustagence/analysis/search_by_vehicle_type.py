import streamlit as st
import pandas as pd
import plotly.express as px


def search_by_vehicle_type(df: pd.DataFrame) -> None:
    """Displays the average claim cost by vehicle type."""

    st.subheader("Claim Cost by Vehicle Type")

    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle",
    }

    st.markdown(
        "This section shows the **average annual claim cost** depending on the vehicle category."
    )

    st.markdown("#### 1. Global relation between claim cost and vehicle type")

    df["vehicle_type"] = df["type_risk"].map(vehicle_type_map)

    mean_df = df.groupby("vehicle_type")["cost_claims_year"].mean().reset_index()

    fig = px.bar(
        mean_df,
        x="vehicle_type",
        y="cost_claims_year",
        title="Prix moyen des sinistres par type de véhicule",
        labels={
            "vehicle_type": "Type de véhicule",
            "claim_amount": "Montant moyen (€)",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 2. Selection of a precise type of vehicle")
    required_cols = ["type_risk", "cost_claims_year"]
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Missing required columns: {required_cols}")
        st.write("Available columns:", df.columns.tolist())
        return

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
    if selected_type == "Agricultural Vehicle":
        st.markdown(
            """
            Those results are a bit strange so here is the global distribution for `Agriculture Vehicle`
            """
        )
        fig = px.scatter(filtered_df, y="cost_claims_year")
        st.plotly_chart(fig, use_container_width=True)

        st.error(
            """
            **Those results are pretty normal because we can say that actually,
            for the people who have agriculture vehicle,
            they have another insurence specifically created for this kind of vehicle.**

            **So actually, those results do not reflect reality**
            """
        )
