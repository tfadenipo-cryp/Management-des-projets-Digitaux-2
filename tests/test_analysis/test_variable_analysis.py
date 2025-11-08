"""Tests for the `variable_analysis` Streamlit component.

This module provides a smoke test ensuring that the function executes without
raising exceptions by mocking Streamlit calls. It follows PEP 8 and PEP 257
conventions for clarity and consistency.
"""

from __future__ import annotations

import pandas as pd
import pytest

# Import directly from package (no sys.path modification)
from src.trustagence.analysis.variable_analysis import variable_analysis


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Return a minimal DataFrame for testing `variable_analysis`."""
    return pd.DataFrame(
        {
            "power": [50, 100, 150],
            "premium": [200, 300, 400],
        }
    )


def test_variable_analysis_executes(
    monkeypatch: pytest.MonkeyPatch, mock_df: pd.DataFrame
) -> None:
    """Smoke test: ensure `variable_analysis` executes without raising errors.

    Streamlit components are patched to prevent actual rendering.
    """

    # Mock Streamlit functions used in the app (safe no-ops)
    monkeypatch.setattr("streamlit.selectbox", lambda *a, **k: "power")
    monkeypatch.setattr("streamlit.plotly_chart", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.write", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.markdown", lambda *a, **k: None, raising=False)

    try:
        variable_analysis(mock_df)  # execute function
    except Exception as exc:  # pragma: no cover - test should not fail
        pytest.fail(f"variable_analysis raised an exception: {exc}")
