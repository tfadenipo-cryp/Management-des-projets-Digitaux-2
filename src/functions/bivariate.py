import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import numpy as np

# Importe les fonctions de logique pure depuis le nouveau module bivariate_logic.py
from functions.logic.bivariate_logic import (
    prep_value_data,
    prep_age_data,
    prep_driver_age_data,
    prep_area_data
)

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