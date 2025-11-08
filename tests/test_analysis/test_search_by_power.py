"""Tests for the `search_by_power` Streamlit component.

This module provides a smoke test ensuring that the function executes without
raising exceptions by mocking Streamlit calls. The style and documentation
follow Python conventions and PEP 257.
"""

from __future__ import annotations

import pandas as pd
import pytest

# Import directly from package (no sys.path modifications)
from src.trustagence.analysis.search_by_power import search_by_power


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a minimal DataFrame for testing search_by_power."""
    return pd.DataFrame(
        {
            "power": [50, 100, 150],
            "cost_claims_year": [200, 400, 600],
        }
    )


def test_search_by_power_executes(
    monkeypatch: pytest.MonkeyPatch, sample_df: pd.DataFrame
) -> None:
    """Smoke test: ensure `search_by_power` executes without errors.

    We mock Streamlit methods typically called by the component to avoid any UI rendering.
    """

    # Mock Streamlit interactions (safe no-ops)
    monkeypatch.setattr("streamlit.selectbox", lambda *a, **k: 100)
    monkeypatch.setattr("streamlit.write", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.markdown", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None, raising=False)

    try:
        search_by_power(sample_df)  # execute function
    except Exception as exc:  # pragma: no cover - test should not fail
        pytest.fail(f"search_by_power raised an exception: {exc}")
