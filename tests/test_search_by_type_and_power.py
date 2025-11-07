import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import search_by_type_and_power  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "type_risk": [1, 2, 3],
            "power": [100, 100, 200],
            "cost_claims_year": [250, 300, 400],
        }
    )


def test_search_by_type_and_power_executes(monkeypatch, sample_df):
    """Ensure search_by_type_and_power executes correctly."""
    monkeypatch.setattr("streamlit.selectbox", lambda label, options: options[0])
    monkeypatch.setattr("streamlit.write", lambda *args, **kwargs: None)
    try:
        search_by_type_and_power(sample_df)
    except Exception as e:
        pytest.fail(f"search_by_type_and_power raised an exception: {e}")
