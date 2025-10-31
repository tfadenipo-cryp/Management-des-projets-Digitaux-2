#Choix des vars pertinentes 
#Var Cible : y = Cost_claims_year (C’est le coût total des sinistres supporté par l’assureur pour ce client sur l’année.)

"""
LASSO Model Script
Ce script reproduit un modèle LASSO (équivalent glmnet en R) pour prédire 'cost_claims_year'.
Il suit la même structure que le script BIC :
    - Chargement via engineering()
    - Prétraitement via preprocess_data_for_modeling()
    - Lasso + CV pour lambda.min
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, lasso_path

# ====== Configuration du projet ======
ROOT_DIR = str(Path(__file__).resolve().parents[2])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from src.functions.engineering import engineering
    from src.models.preprocessing import preprocess_data_for_modeling
except ImportError as e:
    print(f"❌ Import error : {e}")
    sys.exit(1)

TARGET = "cost_claims_year"
CV_FOLDS = 10
RANDOM_STATE = 42


def run_lasso(X, y):
    """Exécute un LASSO avec validation croisée et affiche les résultats."""

    # ========= CHEMIN DES COEFFICIENTS =========
    print("\n--- Calcul du chemin des coefficients (lasso_path) ---")
    alphas, coefs, _ = lasso_path(X, y, alphas=None, max_iter=5000, verbose=False)

    plt.figure(figsize=(8, 6))
    for i in range(coefs.shape[0]):
        plt.plot(np.log10(alphas), coefs[i, :])
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient")
    plt.title("Chemin des coefficients — LASSO (équiv. glmnet)")
    plt.tight_layout()
    plt.savefig("lasso_coef_path.png", bbox_inches="tight")
    plt.close()

    # ========= LassoCV : sélection de lambda.min =========
    print("\n--- Validation croisée (LassoCV) ---")
    lasso_cv = LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE, n_alphas=100, max_iter=5000, n_jobs=-1)
    lasso_cv.fit(X, y)

    lambda_min = lasso_cv.alpha_
    mse_path_mean = lasso_cv.mse_path_.mean(axis=1)
    best_idx = list(lasso_cv.alphas_).index(lambda_min)
    best_mse = float(mse_path_mean[best_idx])
    best_rmse = float(np.sqrt(best_mse))

    plt.figure(figsize=(8, 6))
    plt.plot(np.log10(lasso_cv.alphas_), mse_path_mean, marker="o", ms=3)
    plt.axvline(np.log10(lambda_min), linestyle="--", label=f"lambda.min={lambda_min:.4g}")
    plt.xlabel("log10(lambda)")
    plt.ylabel("MSE (CV)")
    plt.title("Validation croisée — LASSO")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lasso_cv_curve.png", bbox_inches="tight")
    plt.close()

    print("\n=== Résumé LASSO ===")
    print(f"λ.min (alpha*)   : {lambda_min:.6g}")
    print(f"CV MSE (min)     : {best_mse:.6g}")
    print(f"CV RMSE (min)    : {best_rmse:.6g}")
    print(f"Nb de features non nulles : {(lasso_cv.coef_ != 0).sum()} / {len(lasso_cv.coef_)}")

    return lasso_cv


def main():
    """Pipeline complet : engineering → preprocessing → LASSO"""

    print("--- Démarrage du processus LASSO ---")

    # Étape 1 : chargement
    print("1️⃣  Chargement des données via engineering() ...")
    base_df = engineering()
    if base_df is None or base_df.empty:
        print("❌ Aucune donnée chargée. Arrêt.")
        return

    # Étape 2 : prétraitement
    print("2️⃣  Prétraitement avec preprocess_data_for_modeling() ...")
    X_processed, y_target = preprocess_data_for_modeling(base_df, target_column=TARGET)
    if X_processed is None or X_processed.empty or y_target.empty:
        print("❌ Données prétraitées invalides.")
        return

    # Étape 3 : standardisation
    print("3️⃣  Standardisation des features ...")
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_processed)

    # Étape 4 : modèle LASSO
    print("4️⃣  Entraînement du modèle LASSO ...")
    model = run_lasso(X_scaled, y_target)

    # Étape 5 : sortie des coefficients
    print("\n5️⃣  Coefficients non nuls :")
    coef_series = pd.Series(model.coef_, index=X_processed.columns)
    non_zero = coef_series[coef_series != 0].sort_values(key=np.abs, ascending=False)
    print(non_zero.head(30))

    print("\n✅ Figures enregistrées :")
    print(" - lasso_coef_path.png")
    print(" - lasso_cv_curve.png")


if __name__ == "__main__":
    main()
