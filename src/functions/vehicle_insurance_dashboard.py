from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px


# ======================================================================
#                           CHARGEMENT DES DONNEES
# ======================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Charge la base de données d'assurance véhicule depuis le dépôt GitLab.
    Le fichier CSV est supposé être situé à la racine du projet dans :
    data/raw/Motor_vehicle_insurance_data.csv
    """
    project_root = Path(__file__).resolve().parents[0]

    # On remonte jusqu’à trouver data/raw
    while not (project_root / "data" / "raw").exists() and project_root != project_root.parent:
        project_root = project_root.parent

    csv_path = project_root / "data" / "raw" / "Motor_vehicle_insurance_data.csv"
    return pd.read_csv(csv_path, sep=";")

# ======================================================================
#                       AFFICHAGE FORMATÉ DES RÉSULTATS
# ======================================================================

def format_result_display(title: str, results: dict[str, str]) -> None:
    html_text = f"<h3 style='color:#0a84ff; font-family: Arial, sans-serif;'>{title}</h3>"
    for key, value in results.items():
        html_text += f"<h4 style='font-family: Arial, sans-serif;'><b>{key}:</b> {value}</h4>"
    st.markdown(html_text, unsafe_allow_html=True)

# ======================================================================
#                        RECHERCHE PAR PUISSANCE DE VEHICULE
# ======================================================================

def search_by_power(df: pd.DataFrame) -> None:
    power_list: list[float] = sorted(df["Power"].dropna().unique().tolist())
    power_choice = st.selectbox("Select the vehicle's horsepower:", power_list)

    if power_choice is None:
        st.warning("Please select a valid power value.")
        return

    df_clean = df.dropna(subset=["Power", "Cost_claims_year"])
    avg_data = df_clean.groupby("Power", as_index=False)["Cost_claims_year"].mean()
    row = avg_data[avg_data["Power"] == power_choice]

    if not row.empty:
        avg_cost: float = float(row["Cost_claims_year"].values[0])
        format_result_display(f"Vehicle Power: {int(power_choice)} HP", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("Aucun enregistrement trouvé pour cette puissance de véhicule.")

# ======================================================================
#                         RECHERCHE PAR TYPE DE VEHICULE
# ======================================================================

def search_by_vehicle_type(df: pd.DataFrame) -> None:
    vehicle_type_map: dict[int, str] = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle",
    }
    inverse_map: dict[str, int] = {v: k for k, v in vehicle_type_map.items()}

    vehicle_types = list(inverse_map.keys())
    selected_type = st.selectbox("Select a vehicle type:", vehicle_types)

    if not selected_type:
        st.warning("Please select a valid vehicle type.")
        return

    df_clean = df.dropna(subset=["Type_risk", "Cost_claims_year"]).copy()
    df_clean["Vehicle_Type"] = df_clean["Type_risk"].map(vehicle_type_map)
    avg_data = df_clean.groupby("Vehicle_Type", as_index=False)["Cost_claims_year"].mean()
    row = avg_data[avg_data["Vehicle_Type"] == selected_type]

    if not row.empty:
        avg_cost = float(row["Cost_claims_year"].values[0])
        format_result_display(f"Vehicle Type: {selected_type}", {
            "Annual Average Claim Cost": f"${avg_cost:,.2f}",
        })
    else:
        st.info("Aucune donnée trouvée pour ce type de véhicule.")

# ======================================================================
#                   RECHERCHE COMBINÉE TYPE ET PUISSANCE
# ======================================================================

def search_by_type_and_power(df: pd.DataFrame) -> None:
    vehicle_type_map: dict[int, str] = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle",
    }
    inverse_map: dict[str, int] = {v: k for k, v in vehicle_type_map.items()}

    selected_type = st.selectbox("Select vehicle type:", list(inverse_map.keys()))

    if not selected_type:
        st.warning("Please select a valid vehicle type.")
        return

    df_clean = df.dropna(subset=["Type_risk", "Power", "Cost_claims_year"]).copy()
    df_clean["Vehicle_Type"] = df_clean["Type_risk"].map(vehicle_type_map)

    power_list: list[float] = sorted(df_clean[df_clean["Vehicle_Type"] == selected_type]["Power"].unique().tolist())
    selected_power = st.selectbox("Select vehicle horsepower:", power_list)

    if selected_power is None:
        st.warning("Please select a valid power value.")
        return

    filtered_df = df_clean[
        (df_clean["Vehicle_Type"] == selected_type)
        & (df_clean["Power"] == selected_power)
    ]

    if filtered_df.empty:
        st.warning("Aucun enregistrement disponible pour cette combinaison.")
    else:
        avg_claim_cost = float(filtered_df["Cost_claims_year"].mean())
        format_result_display(
            f"Vehicle Type: {selected_type} | Vehicle Power: {int(selected_power)} HP",
            {"Annual Average Claim Cost": f"${avg_claim_cost:,.2f}"}
        )

# ======================================================================
#                      VISUALISATION PLOTLY (ANALYSE SIMPLE)
# ======================================================================

def variable_analysis(df: pd.DataFrame) -> None:
    st.header("📊 Variable Analysis (Plotly)")
    choice: str = st.selectbox("Choose a variable to visualize:", df.columns)
    if choice:
        st.plotly_chart(
            px.histogram(df, x=choice, title=f"Distribution of {choice}"),
            use_container_width=True
        )

# ======================================================================
#                              APPLICATION PRINCIPALE
# ======================================================================

def main() -> None:
    st.set_page_config(page_title="Vehicle Insurance Dashboard", layout="wide")
    df = load_data()

    if "page" not in st.session_state:
        st.session_state.page = "menu"

    if st.session_state.page == "menu":
        st.title("🚗 Vehicle Insurance Data Dashboard")

        choice: str = st.radio(
            "What would you like to explore?",
            ["🔍 Data Analysis", "📊 Variable Visualization"]
        )

        if st.button("Confirm choice"):
            st.session_state.page = "analysis" if choice == "🔍 Data Analysis" else "visualization"
            st.rerun()

    elif st.session_state.page == "analysis":
        st.title("🔍 Data Search Options")
        menu_option: str = st.selectbox("Choose analysis type:", [
            "Search by Vehicle Power",
            "Search by Vehicle Type",
            "Search by Vehicle Type AND Power"
        ])

        if menu_option == "Search by Vehicle Power":
            search_by_power(df)
        elif menu_option == "Search by Vehicle Type":
            search_by_vehicle_type(df)
        elif menu_option == "Search by Vehicle Type AND Power":
            search_by_type_and_power(df)

        st.divider()
        if st.button("⬅️ Back to Main Menu"):
            st.session_state.page = "menu"
            st.rerun()

    elif st.session_state.page == "visualization":
        st.title("📊 Variable Analysis (Plotly)")
        variable_analysis(df)
        st.divider()
        if st.button("⬅️ Back to Main Menu"):
            st.session_state.page = "menu"
            st.rerun()

# ======================================================================
#                             LANCEMENT
# ======================================================================

if __name__ == "__main__":
    main()
