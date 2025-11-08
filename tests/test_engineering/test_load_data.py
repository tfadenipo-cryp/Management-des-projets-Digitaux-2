"""Tests for the `load_data` function.

This module verifies that the data-loading utility returns a valid, non-empty
DataFrame with expected columns and proper data types.
It follows PEP 8 and PEP 257 conventions.
"""

from __future__ import annotations

import pandas as pd
from src.trustagence.engineering.load_data import load_data


def test_load_data_returns_dataframe() -> None:
    """Smoke test: ensure `load_data` returns a valid non-empty DataFrame.

    The test checks for correct type, presence of key columns, and numeric dtypes.
    """

    df = load_data()

    # ---- Basic type validation ----
    assert isinstance(df, pd.DataFrame), "load_data() should return a pandas DataFrame."

    # ---- Non-empty check ----
    assert not df.empty, (
        "The DataFrame is empty — verify that 'data/processed/new_motor_vehicle_insurance_data.csv' exists "
        "and that the delimiter ';' matches the file format."
    )

    # ---- Required columns ----
    expected_cols = {"premium", "cost_claims_year", "power", "value_vehicle"}
    missing_cols = expected_cols - set(df.columns)

    assert not missing_cols, f"Missing required columns: {', '.join(missing_cols)}"

    # ---- Numeric column validation ----
    numeric_cols = ["premium", "cost_claims_year", "value_vehicle"]
    for col in numeric_cols:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(
                df[col]
            ), f"Column '{col}' should contain numeric values."
