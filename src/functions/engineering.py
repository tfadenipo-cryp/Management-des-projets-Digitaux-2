"""
This script has for objectif to manage and custum the database
"""

import pandas as pd
import requests 
import io
import zipfile

def engineering()->  pd.DataFrame:
    """
    This function has for objective to take the data from the website and to transform them into a useful 
    data set where we will be able to use all of the columns to do some analysis and forecast with mathematical methods.
    and for each utilisation. The people who will use the new dataset must execute the command "data = engineering()". 
    Because the dataset will be modified in the future, this function will not remain unchanged.
    
    """
    #get data automatically by the link in the website
    url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5cxyb5fp4f-2.zip"
    resp = requests.get(url, timeout=60)
    #resp.raise_for_status()
    zip_bytes = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_bytes) as z:
        #print("Contenu du zip :", z.namelist())
        #expected path
        csv_path = "Dataset of an actual motor vehicle insurance portfolio/Motor vehicle insurance data.csv"
        
        if csv_path not in z.namelist():
            candidates = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not candidates:
                raise FileNotFoundError("Aucun fichier .csv trouvé dans le ZIP.")
            csv_path = candidates[0]

        with z.open(csv_path) as f:
            # take the data and precise the separation betweein each columns
            data = pd.read_csv(f, sep=None, engine="python")
    
    #make all the columns names lower
    data.columns = data.columns.str.lower()

    #changes the 4 firts to date_time
    colnames_dates = ["date_start_contract","date_last_renewal", "date_next_renewal", "date_birth" ,"date_driving_licence", "date_lapse"]
    for col in colnames_dates :
        data[col] = pd.to_datetime(data[col], format= "%d/%m/%Y")
        data[col] = data[col].dt.date
    
    
    #transformation of the variable "type_fuel" 
    if "type_fuel" in data.columns:
        data["type_fuel"] = (
            data["type_fuel"]
            .map({"P": 1, "D": 2})
            .fillna(0)
            .astype("int8")
        )
        
    #the new way to use the data
    return data
