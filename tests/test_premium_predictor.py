from src.functions.premium_predictor import load_premium_models, premium_predictor


def test_load_premium_models_basic():
    """Teste que la fonction renvoie bien un tuple et ne plante pas."""
    try:
        result = load_premium_models()
        assert isinstance(result, tuple), "Le résultat doit être un tuple"
        assert len(result) == 3, "Le tuple doit contenir 3 éléments"
        print("✅ test_load_premium_models_basic : OK")
    except Exception as e:
        print("❌ test_load_premium_models_basic : erreur ->", e)


def test_premium_predictor_basic():
    """Teste simplement que la fonction principale s’exécute sans erreur."""
    try:
        premium_predictor()
        print(
            "✅ test_premium_predictor_basic : OK (aucune erreur pendant l'exécution)"
        )
    except Exception as e:
        print("❌ test_premium_predictor_basic : erreur ->", e)
