"""
Unified Streamlit Dashboard for Vehicle Insurance Analysis
Combines Idriss' and Nick's analytical dashboards into one interface.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

# ======================================================================
#                           CONFIGURATION
# ======================================================================

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "new_motor_vehicle_insurance_data.csv"

# ======================================================================
#                           DATA LOADING
# ======================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the processed vehicle insurance dataset."""
    try:
        df = pd.read_csv(DATA_PATH, sep=",")
        df.columns = [c.strip().lower() for c in df.columns]  # uniformise les noms de colonnes
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ======================================================================
#                     ORIGINAL ANALYSIS FUNCTIONS (IDRISS)
# ======================================================================

def format_result_display(title: str, results: dict) -> None:
    html_text = f"<h3 style='color:#0a84ff; font-family: Arial, sans-serif;'>{title}</h3>"
    for key, value in results.items():
        html_text += f"<h4 style='font-family: Arial, sans-serif;'><b>{key}:</b> {value}</h4>"
    st.markdown(html_text, unsafe_allow_html=True)

def search_by_power(df: pd.DataFrame) -> None:
    if "power" not in df.columns or "cost_claims_year" not in df.columns:
        st.warning("Required columns missing in dataset.")
        return

    power_list = sorted(df["power"].dropna().unique())
    power_choice = st.selectbox("Select the vehicle's horsepower:", power_list)

    df_clean = df.dropna(subset=["power", "cost_claims_year"])
    avg_data = df_clean.groupby("power")[["cost_claims_year"]].mean().reset_index()
    row = avg_data[avg_data["power"] == power_choice]

    if not row.empty:
        avg_cost = row["cost_claims_year"].values[0]
        format_result_display(f"Vehicle Power: {int(power_choice)} HP", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("No record found for this power.")

def search_by_vehicle_type(df: pd.DataFrame) -> None:
    if "type_risk" not in df.columns or "cost_claims_year" not in df.columns:
        st.warning("Required columns missing in dataset.")
        return

    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle"
    }

    df_clean = df.dropna(subset=["type_risk", "cost_claims_year"])
    df_clean["vehicle_type"] = df_clean["type_risk"].map(vehicle_type_map)
    selected_type = st.selectbox("Select a vehicle type:", df_clean["vehicle_type"].dropna().unique())

    avg_data = df_clean.groupby("vehicle_type")[["cost_claims_year"]].mean().reset_index()
    row = avg_data[avg_data["vehicle_type"] == selected_type]

    if not row.empty:
        avg_cost = row["cost_claims_year"].values[0]
        format_result_display(f"Vehicle Type: {selected_type}", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("No data found for this vehicle type.")

def search_by_type_and_power(df: pd.DataFrame) -> None:
    if not {"type_risk", "power", "cost_claims_year"}.issubset(df.columns):
        st.warning("Required columns missing in dataset.")
        return

    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle"
    }

    df_clean = df.dropna(subset=["type_risk", "power", "cost_claims_year"])
    df_clean["vehicle_type"] = df_clean["type_risk"].map(vehicle_type_map)

    selected_type = st.selectbox("Select vehicle type:", df_clean["vehicle_type"].dropna().unique())
    power_list = sorted(df_clean[df_clean["vehicle_type"] == selected_type]["power"].unique())
    selected_power = st.selectbox("Select vehicle horsepower:", power_list)

    filtered_df = df_clean[
        (df_clean["vehicle_type"] == selected_type)
        & (df_clean["power"] == selected_power)
    ]

    if filtered_df.empty:
        st.warning("No record found for this combination.")
    else:
        avg_claim_cost = filtered_df["cost_claims_year"].mean()
        format_result_display(
            f"{selected_type} | {int(selected_power)} HP",
            {"Average Annual Claim Cost": f"${avg_claim_cost:,.2f}"}
        )

def variable_analysis(df: pd.DataFrame) -> None:
    st.header("📊 Variable Visualization (Plotly)")
    choice = st.selectbox("Choose a variable to visualize:", df.columns)
    st.plotly_chart(px.histogram(df, x=choice, title=f"Distribution of {choice}"), use_container_width=True)

# ======================================================================
#                     ADDITIONAL ANALYSIS (NICK)
# ======================================================================

def prep_value_data(df):
    if not {"value_vehicle", "premium"}.issubset(df.columns):
        return pd.DataFrame()
    df_copy = df[["value_vehicle", "premium"]].dropna()
    df_copy = df_copy[(df_copy["value_vehicle"] > 0) & (df_copy["premium"] > 0)]
    return df_copy

def prep_age_data(df):
    if not {"year_matriculation", "premium"}.issubset(df.columns):
        return pd.DataFrame()
    df_copy = df[["year_matriculation", "premium"]].dropna()
    df_copy["vehicle_age"] = datetime.now().year - df_copy["year_matriculation"]
    return df_copy[df_copy["vehicle_age"] >= 0]

def prep_driver_age_data(df):
    if not {"date_birth", "premium"}.issubset(df.columns):
        return pd.DataFrame()
    df_copy = df[["date_birth", "premium"]].dropna()
    df_copy["date_birth"] = pd.to_datetime(df_copy["date_birth"], errors='coerce')
    df_copy["driver_age"] = (datetime.now() - df_copy["date_birth"]).dt.days / 365.25
    return df_copy[(df_copy["driver_age"] >= 18) & (df_copy["driver_age"] <= 90)]

def prep_area_data(df):
    if not {"area", "premium"}.issubset(df.columns):
        return pd.DataFrame()
    df_copy = df[["area", "premium"]].dropna()
    df_copy["area_type"] = df_copy["area"].apply(lambda x: "Urban" if x == 1 else "Rural")
    return df_copy

def create_value_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df, x="value_vehicle", y="premium", ax=ax, line_kws={'color': 'orange'})
    ax.set_title('Vehicle Value vs Premium')
    return fig

def create_age_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df, x="vehicle_age", y="premium", ax=ax, line_kws={'color': 'orange'})
    ax.set_title('Vehicle Age vs Premium')
    return fig

def create_driver_age_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df, x="driver_age", y="premium", ax=ax, line_kws={'color': 'orange'})
    ax.set_title('Driver Age vs Premium')
    return fig

def create_area_plot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="area_type", y="premium", ax=ax, color='orange')
    ax.set_title('Premium by Geographic Area')
    return fig

def nick_analysis(df):
    """Nick’s additional analyses."""
    analysis = st.selectbox("Choose an analysis:", [
        "Vehicle Value vs Premium",
        "Vehicle Age vs Premium",
        "Driver Age vs Premium",
        "Premium by Area"
    ])

    if analysis == "Vehicle Value vs Premium":
        data = prep_value_data(df)
        fig = create_value_plot(data)
    elif analysis == "Vehicle Age vs Premium":
        data = prep_age_data(df)
        fig = create_age_plot(data)
    elif analysis == "Driver Age vs Premium":
        data = prep_driver_age_data(df)
        fig = create_driver_age_plot(data)
    else:
        data = prep_area_data(df)
        fig = create_area_plot(data)

    st.pyplot(fig)
    if st.checkbox("Show sample data"):
        st.dataframe(data.head())

# ======================================================================
#                             MAIN APP
# ======================================================================

def main():
    st.set_page_config(page_title="Unified Vehicle Insurance Dashboard", layout="wide")
    df = load_data()

    if df.empty:
        st.error("Could not load dataset.")
        return

    st.sidebar.title("🚗 Dashboard Navigation")
    page = st.sidebar.radio("Select section:", [
        "🔍 Idriss' Analyses",
        "📊 Variable Visualization",
        "📈 Nick's Analyses"
    ])

    if page == "🔍 Idriss' Analyses":
        menu_option = st.selectbox("Choose analysis type:", [
            "Search by Vehicle Power",
            "Search by Vehicle Type",
            "Search by Vehicle Type AND Power"
        ])
        if menu_option == "Search by Vehicle Power":
            search_by_power(df)
        elif menu_option == "Search by Vehicle Type":
            search_by_vehicle_type(df)
        else:
            search_by_type_and_power(df)

    elif page == "📊 Variable Visualization":
        variable_analysis(df)

    elif page == "📈 Nick's Analyses":
        nick_analysis(df)

# ======================================================================
#                           EXECUTION GUARD
# ======================================================================

if __name__ == "__main__":
    main()
