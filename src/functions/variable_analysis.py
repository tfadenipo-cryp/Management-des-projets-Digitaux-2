import streamlit as st
import plotly.express as px
import pandas as pd


def variable_analysis(df: pd.DataFrame) -> None:
    st.subheader("Variable Distribution Analysis")

    variable = st.selectbox("Select variable to visualize:", df.columns)
    st.plotly_chart(px.histogram(df, x=variable, title=f"Distribution of {variable}"), use_container_width=True)
