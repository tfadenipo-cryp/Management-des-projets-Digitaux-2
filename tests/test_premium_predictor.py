"""Tests for the `premium_predictor` Streamlit UI.

This module provides a smoke-test that executes the Streamlit function
without raising, by mocking Streamlit widgets and the model loader.
Comments and docstrings are in English and follow standard Python style.
"""

from __future__ import annotations
from src.functions.premium_predictor import premium_predictor
import src.functions.premium_predictor as _premium_module

import pytest
import streamlit as st
import numpy as np


def test_premium_predictor_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: run the Streamlit function and ensure it does not raise.

    We patch the Streamlit API used by the app and the model-loading helper so
    that the test stays fast and deterministic.
    """

    # ---- Basic Streamlit primitives (no-ops) --------------------------------
    monkeypatch.setattr(st, "title", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "warning", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "success", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "write", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "header", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "divider", lambda *a, **k: None)  # no-op

    # ---- Inputs / widgets ----------------------------------------------------
    monkeypatch.setattr(st, "number_input", lambda *a, **k: 10000)  # numeric
    monkeypatch.setattr(st, "slider", lambda *a, **k: 5)  # numeric
    monkeypatch.setattr(st, "selectbox", lambda *a, **k: 1)  # choice index
    monkeypatch.setattr(st, "radio", lambda *a, **k: 0)  # choice index
    monkeypatch.setattr(st, "text_input", lambda *a, **k: "foo")  # text
    monkeypatch.setattr(st, "date_input", lambda *a, **k: None)  # date
    monkeypatch.setattr(
        st, "file_uploader", lambda *a, **k: DummyUploadedFile()
    )  # fake file

    # ---- Containers / layout -------------------------------------------------
    monkeypatch.setattr(st, "expander", lambda *a, **k: DummyCtx())  # ctx
    monkeypatch.setattr(st, "form", lambda *a, **k: DummyCtx())  # ctx
    monkeypatch.setattr(st, "columns", lambda n: [DummyCtx() for _ in range(n)])  # cols
    monkeypatch.setattr(
        st, "container", lambda *a, **k: DummyCtx(), raising=False
    )  # optional

    # ---- Display helpers -----------------------------------------------------
    monkeypatch.setattr(st, "dataframe", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "table", lambda *a, **k: None)  # no-op
    monkeypatch.setattr(st, "pyplot", lambda *a, **k: None, raising=False)  # optional
    monkeypatch.setattr(
        st, "plotly_chart", lambda *a, **k: None, raising=False
    )  # optional
    monkeypatch.setattr(st, "metric", lambda *a, **k: None, raising=False)  # optional

    # ---- Buttons in forms ----------------------------------------------------
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **k: True)  # always submit

    # ---- Session & sidebar ---------------------------------------------------
    monkeypatch.setattr(st, "session_state", {}, raising=False)  # fake state
    monkeypatch.setattr(st, "sidebar", st, raising=False)  # reuse same API

    # ---- Model loader: patch the CORRECT dotted path -------------------------
    # Patch the attribute on the already-imported module object to avoid sys.modules aliasing issues
    monkeypatch.setattr(
        _premium_module,
        "load_premium_models",
        lambda: (DummyPreprocessor(), DummyModel(), ["const", "num_feature"]),
    )

    # Execute and ensure no exception is raised
    try:
        premium_predictor()  # run UI function
    except Exception as exc:  # pragma: no cover - test should not fail here
        pytest.fail(f"premium_predictor raised an exception: {exc}")


class DummyCtx:
    """Trivial context manager used to mock containers/expanders/forms."""

    def __enter__(self) -> "DummyCtx":
        return self  # simply return itself  # inline no-op

    def __exit__(self, *args) -> bool:  # noqa: D401 - simple passthrough
        """Return False to propagate exceptions (default behavior)."""
        return False  # do not suppress exceptions


class DummyUploadedFile:
    """Minimal fake uploaded file returned by `st.file_uploader`."""

    name = "dummy.csv"  # pretend filename

    def read(self) -> bytes:
        """Return a tiny CSV payload as bytes."""
        return b"col1,col2\n1,2\n"  # tiny CSV


class DummyPreprocessor:
    """Fake preprocessor that returns a 1-column numeric design matrix."""

    def transform(self, df):  # type: ignore[no-untyped-def]
        return np.zeros((len(df), 1))  # one numeric feature

    @property
    def named_transformers_(self) -> dict[str, object]:
        """Mimic a ColumnTransformer's components with minimal surface."""

        class Num:  # numeric pipeline mock
            @property
            def feature_names_in_(self) -> list[str]:
                return ["num_feature"]  # single numeric feature name

        class Cat:  # categorical pipeline mock
            def __getitem__(self, key):  # noqa: D401 - minimal mock
                """Return self to allow chained access like ct["cat"].encoder."""
                return self  # chainable

            def get_feature_names_out(self) -> list[str]:
                return []  # no categorical features in this fake

        return {"num": Num(), "cat": Cat()}  # mapping of transformers


class DummyModel:
    """Fake model exposing a `predict` method compatible with scikit-learn."""

    def predict(self, X):  # type: ignore[no-untyped-def]
        """Return a deterministic single-value prediction for any input."""
        return [1234.56]  # constant prediction
