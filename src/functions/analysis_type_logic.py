import pandas as pd

def calculate_avg_cost_by_type(df: pd.DataFrame, selected_type: str) -> float | None:
    """Calculates the average claim cost for a specific vehicle type label."""
    
    vehicle_type_map: dict[int, str] = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle",
    }
    
    df_clean = df.dropna(subset=["type_risk", "cost_claims_year"]).copy()
    
    df_clean["Vehicle_Type_Label"] = df_clean["type_risk"].map(vehicle_type_map)
    
    if selected_type not in df_clean["Vehicle_Type_Label"].unique():
        return None

    avg_data = df_clean.groupby("Vehicle_Type_Label", as_index=False)["cost_claims_year"].mean()
    row = avg_data[avg_data["Vehicle_Type_Label"] == selected_type]

    if not row.empty:
        return float(row["cost_claims_year"].values[0])
    
    return None