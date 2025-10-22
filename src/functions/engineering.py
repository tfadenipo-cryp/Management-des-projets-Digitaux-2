import pandas as pd
from pathlib import Path
import streamlit as st

# ======================================================================
# DATA LOADING & PREPROCESSING PIPELINE
# ======================================================================

@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Loads raw vehicle insurance data."""
    project_root = Path(__file__).resolve().parents[2] 
    data_path = project_root / "data/raw/Motor_vehicle_insurance_data.csv"
    
    try:
        # Using separator ';' based on CSV preview
        return pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        st.error(f"Raw data file not found at: {data_path}")
        return pd.DataFrame()


def clean_and_transform(data: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning and standard transformations."""
    if data.empty:
        return data

    df = data.copy()
    
    # Standardize column names (lowercase)
    df.columns = df.columns.str.lower()

    # Convert date columns
    date_cols = ["date_start_contract", "date_last_renewal", "date_next_renewal", 
                 "date_birth", "date_driving_licence", "date_lapse"]
    
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce').dt.date

    return df


def save_processed_data(df: pd.DataFrame) -> Path:
    """Saves the processed DataFrame to the 'processed' directory."""
    project_root = Path(__file__).resolve().parents[2] 
    output_path = project_root / "data/processed/new_motor_vehicle_insurance_data.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep=';')
    
    return output_path


def get_processed_data() -> pd.DataFrame:
    """Runs the full pipeline: Load -> Clean -> Save -> Return."""
    raw_data = load_raw_data()
    processed_data = clean_and_transform(raw_data)
    if not processed_data.empty:
        save_processed_data(processed_data)
        
    return processed_data


if __name__ == "__main__":
    get_processed_data()
    print("Data processing complete. Processed data saved to data/processed.")