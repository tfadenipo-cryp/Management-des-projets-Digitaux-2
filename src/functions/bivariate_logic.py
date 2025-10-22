import pandas as pd
from datetime import datetime
import numpy as np

# ======================================================================
# DATA PREPARATION FUNCTIONS
# ======================================================================

def prep_value_data(df: pd.DataFrame, value_col: str, premium_col: str) -> pd.DataFrame:
    df_clean = df.dropna(subset=[value_col, premium_col]).copy()
    df_clean = df_clean[pd.to_numeric(df_clean[value_col], errors='coerce').notna()]
    df_clean[value_col] = pd.to_numeric(df_clean[value_col])
    CAP_VALUE = 300000 
    df_clean = df_clean[df_clean[value_col] < CAP_VALUE]
    return df_clean[[value_col, premium_col]].reset_index(drop=True)

def prep_age_data(df: pd.DataFrame, year_col: str, premium_col: str) -> pd.DataFrame:
    df_clean = df.dropna(subset=[year_col, premium_col]).copy()
    df_clean[year_col] = pd.to_numeric(df_clean[year_col], errors='coerce')
    df_clean = df_clean[df_clean[year_col].notna()]
    current_year = datetime.now().year
    df_clean['Vehicle_Age'] = current_year - df_clean[year_col]
    df_clean = df_clean[(df_clean['Vehicle_Age'] >= 0) & (df_clean['Vehicle_Age'] <= 50)]
    return df_clean[['Vehicle_Age', premium_col]].reset_index(drop=True)

def prep_driver_age_data(df: pd.DataFrame, date_col: str, premium_col: str) -> pd.DataFrame:
    df_clean = df.dropna(subset=[date_col, premium_col]).copy()
    if df_clean[date_col].dtype == object:
         df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
    today = datetime.now()
    df_clean['Driver_Age'] = (today - df_clean[date_col].apply(lambda x: datetime.combine(x, datetime.min.time()))).dt.days / 365.25
    df_clean = df_clean[(df_clean['Driver_Age'] >= 18) & (df_clean['Driver_Age'] <= 100)]
    return df_clean[['Driver_Age', premium_col]].reset_index(drop=True)

def prep_area_data(df: pd.DataFrame, area_col: str, premium_col: str) -> pd.DataFrame:
    df_clean = df.dropna(subset=[area_col, premium_col]).copy()
    AREA_MAP = {0: 'Rural', 1: 'Urban'}
    df_clean['Area_Type'] = df_clean[area_col].map(AREA_MAP)
    CAP_PREMIUM = 50000
    df_clean = df_clean[df_clean[premium_col] < CAP_PREMIUM]
    return df_clean[['Area_Type', premium_col]].reset_index(drop=True)