import sys
from pathlib import Path
import pytest
import streamlit as st
from src.functions.premium_predictor import premium_predictor

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


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

    # mock du loader DANS LE MODULE
    monkeypatch.setattr(
        "functions.premium_predictor.load_premium_models",
        # Mock pour renvoyer les 3 objets (preprocessor, model, features)
        lambda: (DummyPreprocessor(), DummyModel(), ["const", "num_feature"]),
    )

    # Suppression des mocks inutiles qui cassaient
    # monkeypatch.setattr(
    #     "functions.premium_predictor_module.sm.load",
    #     lambda *a, **k: DummyModel()
    # )
    # monkeypatch.setattr(
    #     "functions.premium_predictor_module.open",
    #     lambda *a, **k: DummyFeaturesFile(), raising=False
    # )

    try:
        premium_predictor()
    except Exception as e:
        pytest.fail(f"premium_predictor raised an exception: {e}")


class DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DummyPreprocessor:
    def transform(self, df):
        import numpy as np

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


class DummyFeaturesFile:
    def __enter__(self):
        from io import StringIO

        self.buf = StringIO('["const","num_feature"]')
        return self.buf

    def __exit__(self, *a):
        self.buf.close()
