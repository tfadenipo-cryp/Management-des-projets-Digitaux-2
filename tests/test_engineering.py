import unittest
import pandas as pd
from datetime import date
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
from functions.engineering import clean_and_transform 

class TestDataEngineering(unittest.TestCase):
    """Teste la fonction de nettoyage et de transformation des données."""

    def setUp(self):
        """Définit un DataFrame de test avec les dates au format brut."""
        self.raw_data = pd.DataFrame({
            "ID": [1, 2],
            "Date_start_contract": ["05/11/2015", "01/01/2025"], 
            "Date_birth": ["18/03/1975", "INVALID"], 
            "Premium": [222.52, 300.00],
            "Cost_claims_year": [0, 500],
            "UNMODIFIED_COL": [1, 2]
        })
    
    def test_column_rename(self):
        """Vérifie que tous les noms de colonnes sont passés en minuscules."""
        result_df = clean_and_transform(self.raw_data)
        self.assertTrue(all(col == col.lower() for col in result_df.columns))
        self.assertIn('date_start_contract', result_df.columns)

    def test_date_conversion(self):
        """Vérifie que les colonnes de dates sont converties en objets datetime.date."""
        result_df = clean_and_transform(self.raw_data)
        
        self.assertEqual(result_df['date_start_contract'].iloc[0], date(2015, 11, 5))
        
        self.assertEqual(result_df['date_start_contract'].iloc[1], date(2025, 1, 1))

    def test_invalid_date_handling(self):
        """Vérifie que les dates non valides sont traitées (généralement converties en NaT/NaN puis filtrées)."""
        result_df = clean_and_transform(self.raw_data)
        
        self.assertTrue(pd.isna(result_df['date_birth'].iloc[1]))

    def test_data_integrity(self):
        """Vérifie que les colonnes non modifiées conservent leurs données."""
        result_df = clean_and_transform(self.raw_data)
        self.assertEqual(result_df['unmodified_col'].iloc[0], 1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)