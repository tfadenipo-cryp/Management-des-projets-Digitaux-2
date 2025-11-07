# tests/test_premium_predictor_simple.py
import src.functions.premium_predictor as m


def test_load_premium_models_runs_without_crash(monkeypatch):
    """Vérifie juste que la fonction se lance sans planter."""
    # On désactive les erreurs Streamlit
    monkeypatch.setattr(m, "st", type("S", (), {"error": print})())
    result = m.load_premium_models()
    assert isinstance(result, tuple)


def test_premium_predictor_runs(monkeypatch):
    """Test très simple : la fonction principale s’exécute sans erreur."""

    # On remplace Streamlit par un faux objet vide
    class DummyST:
        def __getattr__(self, name):
            return lambda *a, **k: None

    # On crée de faux objets pour éviter les vrais modèles
    dummy_pre = type(
        "Pre",
        (),
        {
            "transform": lambda self, x: x,
            "named_transformers_": {
                "num": type("N", (), {"feature_names_in_": ["x"]})(),
                "cat": {
                    "onehot": type(
                        "O", (), {"get_feature_names_out": lambda self: []}
                    )()
                },
            },
        },
    )()
    dummy_model = type("Model", (), {"predict": lambda self, x: [123]})()

    # On "monkeypatch" (remplace temporairement)
    monkeypatch.setattr(m, "st", DummyST())
    monkeypatch.setattr(
        m, "load_premium_models", lambda: (dummy_pre, dummy_model, ["const", "x"])
    )

    # Si aucune erreur n’est levée, c’est bon
    m.premium_predictor()
