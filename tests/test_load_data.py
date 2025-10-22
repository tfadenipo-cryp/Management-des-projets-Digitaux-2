import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import load_data # noqa: E402


def test_load_data_returns_dataframe():
    """Test that load_data returns a valid DataFrame."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "The DataFrame should not be empty."
    assert "premium" in df.columns or "cost_claims_year" in df.columns, \
        "Expected key columns to exist in the dataset."
