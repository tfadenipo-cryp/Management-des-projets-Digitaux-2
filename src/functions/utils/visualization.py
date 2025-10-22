import pandas as pd
import streamlit as st
import plotly.express as px

# ======================================================================
# GENERAL DATA VISUALIZATION
# ======================================================================

def explore_variables(df: pd.DataFrame) -> None:
    """Displays an interface to visualize the distribution of a selected variable."""
    st.header("Variable Exploration")
    st.markdown("Visualize the distribution of any suitable column in the processed dataset.")
    
    numeric_or_categorical_cols = df.select_dtypes(include=['number', 'object']).columns.tolist()
    
    cols_to_exclude = ['id'] 
    choices = [col for col in numeric_or_categorical_cols if col not in cols_to_exclude and 'date' not in col]

    choice: str = st.selectbox("Choose a variable to visualize:", choices)
    
    if choice:
        try:
            if df[choice].dtype == 'object' and df[choice].nunique() < 50:
                df[choice] = df[choice].astype('category')
                
            fig = px.histogram(df, x=choice, title=f"Distribution of **{choice}**")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create histogram for this variable. Error: {e}")