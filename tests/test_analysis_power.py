import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ======================================================================
# FEATURE 1: SEARCH BY VEHICLE POWER
# ======================================================================

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
from functions.analysis_power import calculate_avg_cost_by_power # <-- IMPORT CORRIGÉ

class TestPowerAnalysis(unittest.TestCase):

    def setUp(self):
        self.df_test = pd.DataFrame({
            "power": [80, 80, 100, 100, np.nan],
            "cost_claims_year": [100, 200, 300, 400, 500] 
        })
        self.df_test.columns = self.df_test.columns.str.lower()
    
    def test_calculation_valid(self):
        result = calculate_avg_cost_by_power(self.df_test, 80)
        self.assertAlmostEqual(result, 150.0, places=2)

    def test_calculation_different_value(self):
        result = calculate_avg_cost_by_power(self.df_test, 100)
        self.assertAlmostEqual(result, 350.0, places=2)

    def test_calculation_non_existent(self):
        result = calculate_avg_cost_by_power(self.df_test, 999)
        self.assertIsNone(result)

    def test_handling_missing_data(self):
        df_nan = pd.DataFrame({
            "power": [50, np.nan],
            "cost_claims_year": [100, 500]
        })
        df_nan.columns = df_nan.columns.str.lower()
        
        result = calculate_avg_cost_by_power(df_nan, 50)
        self.assertAlmostEqual(result, 100.0, places=2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)