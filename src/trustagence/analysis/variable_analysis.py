import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from trustagence.tools.utils import get_name, reright_of


def variable_analysis(df: pd.DataFrame) -> None:
    st.markdown(
        "## Variable Distribution Analysis : choose the variable you want to explore"
    )

    variable = st.selectbox(
        "Select variable to visualize:", [reright_of(var) for var in df.columns]
    )

    st.markdown(
        f"### The type of the variable which is explaining the {variable} is `{df[get_name(variable)].dtype}` and here is an univariate analyse of it."
    )

    types = df[get_name(variable)].dtype
    # title distribution

    if types == "int64" or types == "int8":
        n_mod = len(df[get_name(variable)].unique())

        st.markdown(f"#### 1) Simple distribution of {variable}")
        if n_mod >= 5:
            st.plotly_chart(
                px.histogram(
                    df,
                    x=get_name(variable),
                    title=f"Distribution of {get_name(variable)}",
                ),
                use_container_width=True,
            )
        else:
            fig = px.pie(
                df,
                names=get_name(variable),
                title=f"Distribution of {get_name(variable)}",
                hole=0.3,
            )
            st.plotly_chart(fig)

        st.markdown("#### 2) Statistiques descriptives")
        stats = df[get_name(variable)].describe().to_frame().T
        st.dataframe(stats.style.format("{:.2f}"))

    elif types == "object":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Descriptive statistics")
            st.write(f"First date : {df[get_name(variable)].min()}")
            st.write(f"Last date : {df[get_name(variable)].max()}")
            st.write(f"Number of observation: {len(df)}")
            st.write(
                f" Period covered: {(df[get_name(variable)].max() - df[get_name(variable)].min()).days} days"
            )
        with col2:
            st.subheader("Distribution of the variable")
            st.plotly_chart(
                px.histogram(
                    df,
                    x=get_name(variable),
                    title=f"Distribution of {get_name(variable)}",
                ),
                use_container_width=True,
            )

    elif types == "float64":
        st.markdown(f"#### 1) Descriptive statistics of {variable}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Descriptive statistics")
            stats = df[get_name(variable)].describe().to_frame().T  # résumé statistique
            st.dataframe(stats.style.format("{:.2f}"))  # affichage formaté
        with col2:
            st.markdown("##### Distribution of the variable")
            fig, ax = plt.subplots()
            ax.scatter(range(len(df)), df[get_name(variable)], alpha=0.7)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title("Dots cloud")
            st.pyplot(fig)
