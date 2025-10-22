import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from functions import main_dashboard # noqa: E402


def test_main_dashboard_runs(monkeypatch):
    """Ensure that the main dashboard executes at least its setup without errors."""
    monkeypatch.setattr("streamlit.set_page_config", lambda **kwargs: None)
    monkeypatch.setattr("streamlit.title", lambda text: None)
    monkeypatch.setattr("streamlit.markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr("streamlit.error", lambda *args, **kwargs: None)
    monkeypatch.setattr("streamlit.divider", lambda: None)
    monkeypatch.setattr("streamlit.subheader", lambda *args, **kwargs: None)

    try:
        main_dashboard()
    except Exception as e:
        pytest.fail(f"main_dashboard raised an exception: {e}")
