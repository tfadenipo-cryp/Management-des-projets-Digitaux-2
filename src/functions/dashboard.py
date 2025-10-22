import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


def main_streamlit():
    # Chemin relatif basé sur ce fichier
    data_path = Path(__file__).resolve().parents[2] / "data/raw/Motor_vehicle_insurance_data.csv"
    
    # Lecture de la base
    data = pd.read_csv(data_path, sep=";")
    
    st.title("Analyse des premières variables")

    choix = st.selectbox("Choisis une variable :", data.columns)
    st.plotly_chart(px.histogram(data, x=choix))


# la j'ai mis des graphiques selon ce qu'on veut

# pour faire tourner la code il faut écrire dans le terminal python : "streamlit run [chemin du fichier].py"
