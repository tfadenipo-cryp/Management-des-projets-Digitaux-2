import streamlit as st

# ======================================================================
# GENERAL DISPLAY UTILS
# ======================================================================

def display_results(title: str, results: dict[str, str]) -> None:
    """Displays formatted results as styled metrics in Streamlit."""
    st.markdown(f"### <span style='color:#007bff;'>{title}</span>", unsafe_allow_html=True)
    
    cols = st.columns(len(results))
    for i, (key, value) in enumerate(results.items()):
        with cols[i]:
            st.metric(label=key, value=value)