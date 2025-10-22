import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
from functions.analysis_type_logic import calculate_avg_cost_by_type 

class TestTypeAnalysis(unittest.TestCase):
    """Teste la fonction calculate_avg_cost_by_type."""

    def setUp(self):
        """Définit un DataFrame de test simulé pour le type de risque."""
        self.df_test = pd.DataFrame({
            "type_risk": [1, 1, 3, 3, np.nan],
            "cost_claims_year": [50, 150, 300, 400, 1000] 
        })
        self.df_test.columns = self.df_test.columns.str.lower()
    
    def test_calculation_valid_motorbike(self):
        """Vérifie que le coût moyen est calculé correctement pour Motorbike."""
        result = calculate_avg_cost_by_type(self.df_test, "Motorbike")
        self.assertAlmostEqual(result, 100.0, places=2)

    def test_calculation_valid_car(self):
        """Vérifie que le calcul est correct pour Passenger Car."""
        result = calculate_avg_cost_by_type(self.df_test, "Passenger Car")
        self.assertAlmostEqual(result, 350.0, places=2)

    def test_calculation_non_existent(self):
        """Vérifie que cela retourne None pour un type non trouvé."""
        result = calculate_avg_cost_by_type(self.df_test, "Van")
        self.assertIsNone(result)

    def test_handling_missing_data(self):
        """Vérifie que les lignes avec NaN sont ignorées."""
        result_df = calculate_avg_cost_by_type(self.df_test, "Motorbike")
        self.assertIsNotNone(result_df) 


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)