"""
This scripr has for objectif to manage and custum the database
"""

from pathlib import Path


data_path = Path(__file__).resolve().parents[2] / "data/raw/Motor_vehicle_insurance_data.csv"
data_path.columns