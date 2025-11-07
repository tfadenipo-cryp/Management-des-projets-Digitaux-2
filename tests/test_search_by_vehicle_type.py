import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import search_by_vehicle_type  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame({"type_risk": [1, 2, 3], "cost_claims_year": [100, 300, 500]})


def test_search_by_vehicle_type_executes(monkeypatch, sample_df):
    """Ensure search_by_vehicle_type executes properly."""
    monkeypatch.setattr("streamlit.selectbox", lambda label, options: "Van")
    monkeypatch.setattr("streamlit.write", lambda *args, **kwargs: None)
    try:
        search_by_vehicle_type(sample_df)
    except Exception as e:
        pytest.fail(f"search_by_vehicle_type raised an exception: {e}")
