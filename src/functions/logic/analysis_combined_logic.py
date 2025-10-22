import pandas as pd
import numpy as np


def calculate_avg_cost_combined(df: pd.DataFrame, selected_type: str, selected_power: float) -> float | None:
    """Calculates the average claim cost for a specific vehicle type AND power combination."""
    
    # Mapper les codes de risque comme dans les autres fonctions
    vehicle_type_map = {
        1: "Motorbike",
        2: "Van",
        3: "Passenger Car",
        4: "Agricultural Vehicle"
    }

    # Nettoyer les données essentielles
    df_clean = df.dropna(subset=["type_risk", "cost_claims_year", "power"]).copy()
    df_clean["vehicle_type"] = df_clean["type_risk"].map(vehicle_type_map)

    # Filtrer les données
    filtered = df_clean[
        (df_clean["vehicle_type"] == selected_type) & 
        (df_clean["power"] == selected_power)
    ]

    if not filtered.empty:
        # Calculer la moyenne
        return filtered["cost_claims_year"].mean()
    else:
        return None