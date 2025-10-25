import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import variable_analysis # noqa: E402


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "power": [50, 100, 150],
        "premium": [200, 300, 400]
    })


def test_variable_analysis_executes(monkeypatch, mock_df):
    """Ensure variable_analysis executes without crashing."""
    monkeypatch.setattr("streamlit.selectbox", lambda label, options: "power")
    monkeypatch.setattr("streamlit.plotly_chart", lambda *args, **kwargs: None)
    try:
        variable_analysis(mock_df)
    except Exception as e:
        pytest.fail(f"variable_analysis raised an exception: {e}")

