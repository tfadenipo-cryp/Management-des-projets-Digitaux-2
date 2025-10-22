import pandas as pd

def calculate_avg_cost_by_power(df: pd.DataFrame, power_choice: float) -> float | None:
    """Calculates the average claim cost for a specific vehicle power."""
    
    df_clean = df.dropna(subset=["power", "cost_claims_year"])
    
    avg_data = df_clean.groupby("power", as_index=False)["cost_claims_year"].mean()
    row = avg_data[avg_data["power"] == power_choice]

    if not row.empty:
        return float(row["cost_claims_year"].values[0])
    
    return None