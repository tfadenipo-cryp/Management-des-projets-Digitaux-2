import pandas as pd
import streamlit as st
import plotly.express as px

#On lit la base (chemin à changer pour que tout le monde puisse charger la base de données)
data = pd.read_csv("../data/processed//Motor_vehicle_insurance_data.csv", sep = ";")

st.title("Analyse des premières variables")

choix = st.selectbox("Choisis une variable :", data.columns)
st.plotly_chart(px.histogram(data, x=choix))
#la j'ai mis des graphiques selon ce qu'on veut

# pour faire tourner la code il faut écrire dans le terminal python : "streamlit run [chemin du fichier].py"



