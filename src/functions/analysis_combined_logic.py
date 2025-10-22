import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ======================================================================
# CONFIGURATION ET IMPORTS
# ======================================================================

# Assurez-vous que ce chemin est inclus pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
from functions.analysis_combined_logic import calculate_avg_cost_combined 

class TestCombinedAnalysis(unittest.TestCase):
    """Teste la logique de calcul du coût moyen pour une combinaison Type ET Puissance."""

    def setUp(self):
        """Définit un DataFrame de test simulé."""
        self.df_test = pd.DataFrame({
            "type_risk": [1, 1, 3, 3, 3], 
            "power": [50, 80, 80, 100, 100],
            "cost_claims_year": [100, 200, 300, 400, 600] 
        })
        self.df_test.columns = self.df_test.columns.str.lower()
    
    def test_calculation_valid(self):
        """Vérifie le coût moyen pour 'Passenger Car' et 100 HP."""
        # Expected: (400 + 600) / 2 = 500
        result = calculate_avg_cost_combined(self.df_test, "Passenger Car", 100)
        self.assertAlmostEqual(result, 500.0, places=2)

    def test_calculation_single_record(self):
        """Vérifie le coût moyen pour 'Motorbike' et 50 HP (un seul enregistrement)."""
        # Expected: 100
        result = calculate_avg_cost_combined(self.df_test, "Motorbike", 50)
        self.assertAlmostEqual(result, 100.0, places=2)

    def test_combination_not_found(self):
        """Vérifie que None est retourné pour une combinaison inexistante."""
        # Passenger Car + 50 HP n'existe pas
        result = calculate_avg_cost_combined(self.df_test, "Passenger Car", 50)
        self.assertIsNone(result)

    def test_type_non_existent(self):
        """Vérifie que None est retourné pour un type non mappé ou inexistant dans les données."""
        result = calculate_avg_cost_combined(self.df_test, "Truck", 100)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)