from pathlib import Path
import pandas as pd
import streamlit as st
from functions.engineering import engineering


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the processed vehicle insurance dataset.
    The function automatically detects the correct path for any environment.
    """
    
    #old way to load the data
    project_root = Path(__file__).resolve()
    while not (project_root / "data" / "processed").exists() and project_root != project_root.parent:
        project_root = project_root.parent

    csv_path = project_root / "data" / "processed" / "new_motor_vehicle_insurance_data.csv"

    try:
        df = pd.read_csv(csv_path, sep=",")
    except FileNotFoundError:
        st.error(f"❌ File not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error while loading data: {e}")
        return pd.DataFrame()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )



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