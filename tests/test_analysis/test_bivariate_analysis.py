"""Tests for the `bivariate_analysis` Streamlit component.

This module provides a smoke test to ensure the function executes without
raising by mocking Streamlit calls used inside the UI helper.
All comments/docstrings are in English and conventions follow standard Python style.
"""

import pandas as pd
import pytest

# Import directly from the package (no sys.path hacks)
from src.functions.bivariate_analysis import bivariate_analysis


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Return a tiny DataFrame sufficient for a bivariate plot."""
    return pd.DataFrame(
        {
            "value_vehicle": [10000, 20000, 30000],
            "premium": [500, 700, 900],
            "year_matriculation": [2015, 2018, 2020],
            "date_birth": ["1980-01-01", "1990-06-01", "1985-03-15"],
            "area": [1, 2, 1],
        }
    )


def test_bivariate_analysis_executes(
    monkeypatch: pytest.MonkeyPatch, mock_df: pd.DataFrame
) -> None:
    """Smoke test: run `bivariate_analysis` and ensure it does not raise.

    We patch a few Streamlit primitives that are typically used in this
    function so the test remains fast and deterministic.
    """

    # Widgets / outputs (no-ops)
    monkeypatch.setattr(
        "streamlit.selectbox", lambda *a, **k: "Vehicle Value vs Premium"
    )
    monkeypatch.setattr("streamlit.pyplot", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.write", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.markdown", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None, raising=False)

    try:
        bivariate_analysis(mock_df)  # run the UI helper
    except Exception as exc:  # pragma: no cover - should not trigger
        pytest.fail(f"bivariate_analysis raised an exception: {exc}")
