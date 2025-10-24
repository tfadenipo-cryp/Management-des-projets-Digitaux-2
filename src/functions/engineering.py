"""
This scripr has for objectif to manage and custum the database
"""
import pandas as pd
from pathlib import Path



data_path = Path(__file__).resolve().parents[2] / "data/raw/Motor_vehicle_insurance_data.csv"
data = pd.read_csv(data_path, sep=";")

#make all the columns names lower
data.columns = data.columns.str.lower()

#Look at the types of our variables


#changes the 4 firts to date_time

colnames_dates = ["date_start_contract","date_last_renewal", "date_next_renewal", "date_birth" ,"date_driving_licence", "date_lapse"]
for col in colnames_dates :
    data[col] = pd.to_datetime(data[col], format="%d/%m/%Y")
    data[col] = data[col].dt.date
    
    

print(len(data[data["cost_claims_year"]!=0]),len(data),"donc : ",100*len(data[data["cost_claims_year"]!=0])/len(data),"%\n19000 lignes pour faire le modele")





#download gthe new_data.
new_fata_path = Path(__file__).resolve().parents[2] / "data/processed/new_motor_vehicle_insurance_data.csv"
#data.to_csv(new_fata_path, index=False)
