from pathlib import Path
import pandas as pd
import streamlit as st
from engineering import engineering

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the processed vehicle insurance dataset.
    Automatically detects the correct path and standardizes column names.
    """
<<<<<<< HEAD
    
    #old way to load the data
=======

>>>>>>> 05bb5a24fe5747daad24eb91f7a3099066c6a828
    project_root = Path(__file__).resolve()
    while not (project_root / "data" / "processed").exists() and project_root != project_root.parent:
        project_root = project_root.parent

    csv_path = project_root / "data" / "processed" / "new_motor_vehicle_insurance_data.csv"

    try:
<<<<<<< HEAD
        df = pd.read_csv(csv_path, sep=",")
=======
        # ✅ Back to semicolon separator
        df = pd.read_csv(csv_path, sep=";")
>>>>>>> 05bb5a24fe5747daad24eb91f7a3099066c6a828
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

<<<<<<< HEAD


    #other method to loads the data with the new file engineering (we do not have to download the data now
    # and if the wesite wants to add new row to their data, the project will follow the changes)
    data = engineering()
    
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    
    return data
=======
    # ✅ Fix plural or inconsistent names
    rename_map = {
        "claim_costs_year": "cost_claims_year",  # normalize
    }
    df.rename(columns=rename_map, inplace=True)

    return df
>>>>>>> 05bb5a24fe5747daad24eb91f7a3099066c6a828
