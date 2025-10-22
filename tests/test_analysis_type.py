import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
from functions.analysis_type_logic import calculate_avg_cost_by_type 

class TestTypeAnalysis(unittest.TestCase):

    def setUp(self):
        self.df_test = pd.DataFrame({
            "type_risk": [1, 1, 3, 3, np.nan],
            "cost_claims_year": [50, 150, 300, 400, 1000] 
        })
        self.df_test.columns = self.df_test.columns.str.lower()
    
    def test_calculation_valid_motorbike(self):
        result = calculate_avg_cost_by_type(self.df_test, "Motorbike")
        self.assertAlmostEqual(result, 100.0, places=2)

    def test_calculation_valid_car(self):
        result = calculate_avg_cost_by_type(self.df_test, "Passenger Car")
        self.assertAlmostEqual(result, 350.0, places=2)

    def test_calculation_non_existent(self):
        result = calculate_avg_cost_by_type(self.df_test, "Truck")
        self.assertIsNone(result)

    def test_handling_missing_data(self):
        result_df = calculate_avg_cost_by_type(self.df_test, "Motorbike")
        self.assertIsNotNone(result_df) 


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)