import pandas as pd
import streamlit as st
import plotly.express as px
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


# ======================================================================
# STREAMLIT INTEGRATION FUNCTION 
# ======================================================================

def render_bivariate_analysis(df: pd.DataFrame) -> None:
    """Orchestrates the display of Bivariate Analysis results."""
    st.header("Bivariate Analysis: Key Drivers")
    st.markdown("Explore how major factors impact the insurance premium.")

    df.columns = df.columns.str.lower()

    df_value = prep_value_data(df, 'value_vehicle', 'premium')
    df_age = prep_age_data(df, 'year_matriculation', 'premium')
    df_driver_age = prep_driver_age_data(df, 'date_birth', 'premium')
    df_area = prep_area_data(df, 'area', 'premium')

    tab_value, tab_age, tab_driver, tab_area = st.tabs([
        "Vehicle Value vs Premium", 
        "Vehicle Age vs Premium", 
        "Driver Age vs Premium", 
        "Premium by Area"
    ])
    
    with tab_value:
        st.subheader("Vehicle Value vs. Premium")
        st.plotly_chart(px.scatter(df_value, x='value_vehicle', y='premium', trendline="ols"), use_container_width=True)

    with tab_age:
        st.subheader("Vehicle Age vs. Premium")
        st.plotly_chart(px.box(df_age, x='Vehicle_Age', y='premium'), use_container_width=True)

    with tab_driver:
        st.subheader("Driver Age vs. Premium")
        df_driver_age['Age_Group'] = pd.cut(df_driver_age['Driver_Age'], bins=10)
        df_driver_age['Age_Group'] = df_driver_age['Age_Group'].astype(str) 
        
        st.plotly_chart(px.box(df_driver_age, x='Age_Group', y='premium'), use_container_width=True)
        
    with tab_area:
        st.subheader("Premium by Area")
        avg_premium = df_area.groupby('Area_Type')['premium'].mean().reset_index()
        st.plotly_chart(px.bar(avg_premium, x='Area_Type', y='premium', title="Average Premium by Area"), use_container_width=True)