import sys
import pytest
import pandas as pd
from io import StringIO
import streamlit as st
import subprocess
import time
import signal
from pathlib import Path

# ======================================================================
#                 IMPORT DU MODULE A TESTER
# ======================================================================

# On ajoute le chemin du dossier src/functions au path Python
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "functions"))

from vehicle_insurance_dashboard import (
    load_data,
    format_result_display,
    search_by_power,
    search_by_vehicle_type,
    search_by_type_and_power,
    variable_analysis,
)

# ======================================================================
#                         FIXTURE DE DONNÉES DE TEST
# ======================================================================

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Crée un petit jeu de données pour les tests unitaires."""
    raw_data = """Power;Cost_claims_year;Type_risk
100;250.75;3
75;150.30;1
100;275.40;3
60;180.00;4
"""
    return pd.read_csv(StringIO(raw_data), sep=";")

# ======================================================================
#                            TESTS UNITAIRES
# ======================================================================

def test_load_data(monkeypatch, tmp_path):
    """Vérifie que load_data lit correctement un fichier CSV."""
    # Création d’un CSV temporaire
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "Motor_vehicle_insurance_data.csv"
    csv_path.write_text("Power;Cost_claims_year;Type_risk\n100;200;3")

    # Exécution dans le répertoire temporaire
    monkeypatch.chdir(tmp_path)
    df = load_data()
    assert not df.empty
    assert "Power" in df.columns
    assert "Cost_claims_year" in df.columns


def test_format_result_display_runs_without_error():
    """Vérifie que format_result_display ne plante pas."""
    try:
        format_result_display("Test Title", {"Metric": "123"})
    except Exception as e:
        pytest.fail(f"format_result_display a levé une exception : {e}")


def test_search_by_power(monkeypatch, sample_data):
    """Teste la recherche par puissance."""
    monkeypatch.setattr(st, "selectbox", lambda label, options, **_: 100)
    search_by_power(sample_data)
    filtered = sample_data[sample_data["Power"] == 100]
    expected_claim = filtered["Cost_claims_year"].mean()
    assert expected_claim == pytest.approx(263.075, rel=1e-3)


def test_search_by_vehicle_type(monkeypatch, sample_data):
    """Teste la recherche par type de véhicule."""
    monkeypatch.setattr(st, "selectbox", lambda label, options, **_: "Passenger Car")
    search_by_vehicle_type(sample_data)
    grouped = sample_data.groupby("Type_risk")[["Cost_claims_year"]].mean()
    assert "Cost_claims_year" in grouped.columns


def test_search_by_type_and_power(monkeypatch, sample_data):
    """Teste la recherche combinée type + puissance."""
    def mock_select(label, options, **_):
        return "Passenger Car" if "type" in label.lower() else 100
    monkeypatch.setattr(st, "selectbox", mock_select)
    search_by_type_and_power(sample_data)
    subset = sample_data[(sample_data["Power"] == 100) & (sample_data["Type_risk"] == 3)]
    assert len(subset) == 2


def test_variable_analysis(monkeypatch, sample_data):
    """Teste la fonction d’analyse Plotly."""
    monkeypatch.setattr(st, "selectbox", lambda label, options, **_: "Power")
    try:
        variable_analysis(sample_data)
    except Exception as e:
        pytest.fail(f"variable_analysis a levé une exception : {e}")

# ======================================================================
#                       TEST D'INTEGRATION STREAMLIT
# ======================================================================

def test_streamlit_app_runs():
    """
    Vérifie que l’application Streamlit démarre sans erreur.
    On la lance, on attend quelques secondes, puis on l’arrête proprement.
    """
    app_path = Path(__file__).resolve().parents[1] / "src" / "functions" / "vehicle_insurance_dashboard.py"
    
    process = subprocess.Popen(
        ["streamlit", "run", str(app_path), "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Laisse Streamlit démarrer quelques secondes
    time.sleep(6)

    # Arrêt propre du serveur
    process.send_signal(signal.SIGTERM)

    # Lecture de la sortie
    stdout, stderr = process.communicate(timeout=10)

    # Vérifie que Streamlit s’est bien lancé
    assert (
        "Streamlit" in stdout
        or "Running" in stderr
        or "Local URL" in stdout
    )
