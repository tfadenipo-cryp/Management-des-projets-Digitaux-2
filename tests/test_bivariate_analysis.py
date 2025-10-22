import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import bivariate_analysis


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "value_vehicle": [10000, 20000, 30000],
        "premium": [500, 700, 900],
        "year_matriculation": [2015, 2018, 2020],
        "date_birth": ["1980-01-01", "1990-06-01", "1985-03-15"],
        "area": [1, 2, 1],
    })


def test_bivariate_analysis_executes(monkeypatch, mock_df):
    """Ensure that bivariate_analysis runs without errors."""
    monkeypatch.setattr("streamlit.selectbox", lambda label, options: "Vehicle Value vs Premium")
    monkeypatch.setattr("streamlit.pyplot", lambda *args, **kwargs: None)
    try:
        bivariate_analysis(mock_df)
    except Exception as e:
        pytest.fail(f"bivariate_analysis raised an exception: {e}")
