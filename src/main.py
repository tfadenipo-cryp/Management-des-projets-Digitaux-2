"""
Final launcher for the Vehicle Insurance Streamlit Dashboard
------------------------------------------------------------
This file serves as the single entry point for the full Streamlit app.

To launch the dashboard, run:

    streamlit run src/main_py.py
"""

from trustagence.main_dashboard import run_dashboard  # type: ignore


if __name__ == "__main__":
    run_dashboard()
