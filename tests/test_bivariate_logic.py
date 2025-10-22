import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
from functions.bivariate_logic import (
    prep_value_data,
    prep_age_data,
    prep_driver_age_data,
    prep_area_data
)

class TestBivariateLogic(unittest.TestCase):

    def test_prep_value_data(self):
        data = {
            'value_vehicle': [10000, 20000, 999999, 'invalid', np.nan],
            'premium': [500, 800, 9999, 600, 700]
        }
        df = pd.DataFrame(data)
        result_df = prep_value_data(df, 'value_vehicle', 'premium')
                
        self.assertEqual(len(result_df), 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(result_df['value_vehicle']))
        self.assertTrue(result_df['value_vehicle'].max() < 300000) 

    def test_prep_age_data(self):
        current_year = datetime.now().year
        data = {
            'year_matriculation': [current_year - 5, current_year - 15, current_year + 5, 'invalid', np.nan, current_year - 10],
            'premium': [500, 400, 800, 600, 700, 99999]
        }
        df = pd.DataFrame(data)
        result_df = prep_age_data(df, 'year_matriculation', 'premium')
        self.assertEqual(len(result_df), 3) 
        self.assertIn('Vehicle_Age', result_df.columns)
        self.assertTrue(all(result_df['Vehicle_Age'] >= 0))
        self.assertIn(5, result_df['Vehicle_Age'].values)
        self.assertIn(15, result_df['Vehicle_Age'].values)
        self.assertIn(10, result_df['Vehicle_Age'].values)
        
    def test_prep_driver_age_data(self):
        current_date = datetime.now().date()
        date_30_years_ago = current_date.replace(year=current_date.year - 30)
        
        data = {
            'date_birth': [date_30_years_ago, datetime.now().date().replace(year=current_date.year - 17), datetime.now().date().replace(year=current_date.year - 100), None, np.nan],
            'premium': [500, 1200, 900, 600, 700]
        }
        df = pd.DataFrame(data)
        result_df = prep_driver_age_data(df, 'date_birth', 'premium')
        
        self.assertEqual(len(result_df), 2)
        self.assertIn('Driver_Age', result_df.columns)
        self.assertTrue(all(age >= 18 for age in result_df['Driver_Age']))
        
    def test_prep_area_data(self):
        data = {
            'area': [0, 1, 0, 1, np.nan, 0, 999], 
            'premium': [400, 700, 350, 750, 500, 99999, 600]
        }
        df = pd.DataFrame(data)
        result_df = prep_area_data(df, 'area', 'premium')
        
        self.assertEqual(len(result_df), 4) 
        self.assertIn('Area_Type', result_df.columns)
        self.assertIn('Rural', result_df['Area_Type'].values)
        self.assertIn('Urban', result_df['Area_Type'].values)
        self.assertTrue(result_df['premium'].max() < 50000)
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)