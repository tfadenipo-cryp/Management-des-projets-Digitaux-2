"""Tests for the `main_dashboard` Streamlit page.

This file provides a minimal smoke test ensuring the main dashboard page
runs its setup without raising. Comments and docstrings follow PEP 257
and the overall structure matches Python testing conventions.
"""

from __future__ import annotations
import pytest

# Import directly from package structure (no sys.path hacks)
from src.functions.main_dashboard import main_dashboard


def test_main_dashboard_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: ensure that `main_dashboard` executes without errors.

    This test mocks key Streamlit methods typically called during page setup.
    It does not check rendering but only verifies that no exceptions occur.
    """

    # Mock Streamlit UI calls (safe no-ops)
    monkeypatch.setattr("streamlit.set_page_config", lambda **kwargs: None)
    monkeypatch.setattr("streamlit.title", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.markdown", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.error", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.divider", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.subheader", lambda *a, **k: None)
    monkeypatch.setattr("streamlit.warning", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("streamlit.success", lambda *a, **k: None, raising=False)

    try:
        main_dashboard()  # execute Streamlit page setup
    except Exception as exc:  # pragma: no cover - should not trigger
        pytest.fail(f"main_dashboard raised an exception: {exc}")
