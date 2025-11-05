import pandas as pd
import streamlit as st
from .engineering import engineering


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the processed vehicle insurance dataset.
    The function automatically detects the correct path for any environment.
    """

    #Method to loads the data with the new file engineering (we do not have to download the data now
    # and if the wesite wants to add new row to their data, the project will follow the changes)
    try:
        data = engineering()
    except FileNotFoundError:
        st.error("❌ File not found: the data are maybe delated ")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Error while loading data: {e}")
        return pd.DataFrame()
    
    
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    
    return data