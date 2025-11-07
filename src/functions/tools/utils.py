import streamlit as st


def format_result_display(title: str, results: dict):
    html = f"<h3 style='color:#0a84ff;'>{title}</h3>"
    for k, v in results.items():
        html += f"<p><b>{k}:</b> {v}</p>"
    st.markdown(html, unsafe_allow_html=True)
