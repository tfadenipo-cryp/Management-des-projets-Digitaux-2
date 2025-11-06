import sys
from pathlib import Path
import importlib
import pytest
import streamlit as st
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

premium_module = importlib.import_module("functions.premium_predictor")

class DummyCtx:
    def __enter__(self): return self
    def __exit__(self, *args): return False

class DummyPreprocessor:
    def transform(self, df):
        return np.zeros((len(df), 1))

    @property
    def named_transformers_(self):
        class Num:
            @property
            def feature_names_in_(self):
                return ["num_feature"]
        class Cat:
            def __getitem__(self, key):
                return self
            def get_feature_names_out(self):
                return []
        return {"num": Num(), "cat": Cat()}

class DummyModel:
    def predict(self, X):
        return [1234.56]

def test_premium_predictor_executes(monkeypatch):
    monkeypatch.setattr(st, "title", lambda *a, **k: None)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "expander", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "form", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "columns", lambda n: [DummyCtx(), DummyCtx()])
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st, "number_input", lambda *a, **k: 10000)
    monkeypatch.setattr(st, "slider", lambda *a, **k: 5)
    monkeypatch.setattr(st, "selectbox", lambda *a, **k: 1)
    monkeypatch.setattr(st, "radio", lambda *a, **k: 0)
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **k: True)
    monkeypatch.setattr(st, "info", lambda *a, **k: None)
    monkeypatch.setattr(st, "dataframe", lambda *a, **k: None)

    monkeypatch.setattr(
        premium_module, "_load_premium_models_raw",
        lambda: (DummyPreprocessor(), DummyModel(), ["const", "num_feature"])
    )

    try:
        premium_module.premium_predictor()
    except Exception as e:
        pytest.fail(f"premium_predictor raised an exception: {e}")
