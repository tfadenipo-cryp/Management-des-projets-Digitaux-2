import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import search_by_power  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame({"power": [50, 100, 150], "cost_claims_year": [200, 400, 600]})


def test_search_by_power_executes(monkeypatch, sample_df):
    """Ensure search_by_power executes without raising errors."""
    monkeypatch.setattr("streamlit.selectbox", lambda label, options: 100)
    monkeypatch.setattr("streamlit.write", lambda *args, **kwargs: None)
    try:
        search_by_power(sample_df)
    except Exception as e:
        pytest.fail(f"search_by_power raised an exception: {e}")
