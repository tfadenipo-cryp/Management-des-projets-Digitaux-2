from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the processed vehicle insurance dataset.
    Automatically detects the correct path and standardizes column names.
    """

    project_root = Path(__file__).resolve()
    while not (project_root / "data" / "processed").exists() and project_root != project_root.parent:
        project_root = project_root.parent

    csv_path = project_root / "data" / "processed" / "new_motor_vehicle_insurance_data.csv"

    try:
        # ✅ Back to semicolon separator
        df = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError:
        st.error(f"❌ File not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error while loading data: {e}")
        return pd.DataFrame()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    # ✅ Fix plural or inconsistent names
    rename_map = {
        "claim_costs_year": "cost_claims_year",  # normalize
    }
    df.rename(columns=rename_map, inplace=True)

    return df
