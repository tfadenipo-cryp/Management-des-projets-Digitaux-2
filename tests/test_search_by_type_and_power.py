"""Tests for the `search_by_type_and_power` Streamlit component.

This module provides a smoke test ensuring the function executes without
raising by mocking Streamlit interactions. Comments/docstrings are in English
and follow standard Python conventions.
"""

from __future__ import annotations

import pandas as pd
import pytest

# Import directly from package (no sys.path modifications)
from src.functions.search_by_type_and_power import search_by_type_and_power


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a minimal DataFrame for the component under test."""
    return pd.DataFrame(
        {
            "type_risk": [1, 2, 3],
            "power": [100, 100, 200],
            "cost_claims_year": [250, 300, 400],
        }
    )


def test_search_by_type_and_power_executes(
    monkeypatch: pytest.MonkeyPatch, sample_df: pd.DataFrame
) -> None:
    """Smoke test: ensure `search_by_type_and_power` executes without errors.

    We patch Streamlit primitives used by the component to avoid UI side effects.
    """

    # Widgets / outputs (safe no-ops)
    monkeypatch.setattr("streamlit.selectbox", lambda *a, **k: 1)
    monkeypatch.setattr("streamlit.write", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.markdown", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None, raising=False)

    try:
        search_by_type_and_power(sample_df)  # execute
    except Exception as exc:  # pragma: no cover - should not trigger
        pytest.fail(f"search_by_type_and_power raised an exception: {exc}")
