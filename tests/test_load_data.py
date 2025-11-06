import pandas as pd
from functions import load_data  # ✅ imported from src/functions/__init__.py

def test_load_data_returns_dataframe():
    """Test that load_data loads the dataset correctly and returns a valid DataFrame."""
    
    df = load_data()

    # Basic type check
    assert isinstance(df, pd.DataFrame), "❌ load_data() should return a pandas DataFrame."

    # DataFrame not empty
    assert not df.empty, (
        "❌ The DataFrame is empty — check that 'data/processed/new_motor_vehicle_insurance_data.csv' exists "
        "and that the separator (;) matches the file format."
    )

    # Check column names
    expected_cols = {"premium", "cost_claims_year", "power", "value_vehicle"}
    missing_cols = expected_cols - set(df.columns)

    # If missing columns, show them in the error message
    assert not missing_cols, f"⚠️ Missing required columns: {', '.join(missing_cols)}"

    # Optional: check that numeric columns are indeed numeric
    numeric_cols = ["premium", "cost_claims_year", "value_vehicle"]
    for col in numeric_cols:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"⚠️ Column '{col}' should be numeric."
