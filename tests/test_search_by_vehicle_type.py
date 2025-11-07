"""Tests for the `search_by_vehicle_type` Streamlit component.

This module provides a smoke test to ensure the function executes without
raising exceptions by mocking Streamlit interactions. The code follows
standard Python conventions and PEP 257 documentation style.
"""

from __future__ import annotations

import pandas as pd
import pytest

# Import directly from package (no sys.path manipulation)
from src.functions.search_by_vehicle_type import search_by_vehicle_type


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a minimal DataFrame for testing the component."""
    return pd.DataFrame(
        {
            "type_risk": [1, 2, 3],
            "cost_claims_year": [100, 300, 500],
        }
    )


def test_search_by_vehicle_type_executes(
    monkeypatch: pytest.MonkeyPatch, sample_df: pd.DataFrame
) -> None:
    """Smoke test: ensure `search_by_vehicle_type` executes without errors.

    This test mocks common Streamlit functions to avoid UI rendering.
    """

    # Mock Streamlit calls (safe no-ops)
    monkeypatch.setattr("streamlit.selectbox", lambda *a, **k: "Van")
    monkeypatch.setattr("streamlit.write", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.markdown", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None, raising=False)

    try:
        search_by_vehicle_type(sample_df)  # run function
    except Exception as exc:  # pragma: no cover - test should not fail
        pytest.fail(f"search_by_vehicle_type raised an exception: {exc}")
