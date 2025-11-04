"""
LASSO Model Script — Prediction of 'cost_claims_year'
-------------------------------------------------
Ce script applique un modèle LASSO pour prédire le coût annuel
des sinistres supportés par l’assureur ('cost_claims_year'),
en appliquant une transformation logarithmique pour corriger
l’asymétrie des montants.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, lasso_path

# ====== Configuration ======
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


# ====== Fonction d'entraînement ======
def run_lasso(X, y):
    """Exécute un LASSO avec validation croisée et affiche les résultats."""
    print("\n--- Calcul du chemin des coefficients (lasso_path) ---")
    alphas, coefs, _ = lasso_path(X, y, alphas=None, max_iter=5000, verbose=False)

    plt.figure(figsize=(8, 6))
    for i in range(coefs.shape[0]):
        plt.plot(np.log10(alphas), coefs[i, :])
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficient")
    plt.title("Chemin des coefficients — LASSO (log-cost_claims_year)")
    plt.tight_layout()
    plt.savefig("lasso_coef_path_cost_claims.png", bbox_inches="tight")
    plt.close()

    print("\n--- Validation croisée (LassoCV) ---")
    lasso_cv = LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE,
                       n_alphas=100, max_iter=5000, n_jobs=-1)
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
    plt.title("Validation croisée — LASSO (cost_claims_year)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lasso_cv_curve_cost_claims.png", bbox_inches="tight")
    plt.close()

    print("\n=== Résumé LASSO ===")
    print(f"λ.min (alpha*)   : {lambda_min:.6g}")
    print(f"CV MSE (min)     : {best_mse:.6g}")
    print(f"CV RMSE (min)    : {best_rmse:.6g}")
    print(f"Nb de features non nulles : {(lasso_cv.coef_ != 0).sum()} / {len(lasso_cv.coef_)}")

    return lasso_cv


# ====== Pipeline principal ======
def main():
    """Pipeline complet : engineering → preprocessing → LASSO sur log(cost_claims_year)"""

    print("--- Démarrage du processus LASSO (cost_claims_year) ---")

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

    print(f"\nDistribution initiale de {TARGET} :")
    print(y_target.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))

    # Étape 3 : transformation log
    print("3️⃣  Transformation log(1 + y) pour corriger l’asymétrie ...")
    y_log = np.log1p(y_target)

    # Étape 4 : standardisation
    print("4️⃣  Standardisation des features ...")
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_processed)

    # Étape 5 : entraînement du modèle
    print("5️⃣  Entraînement du modèle LASSO (log-cost_claims_year) ...")
    model = run_lasso(X_scaled, y_log)

    # Étape 6 : coefficients
    print("\n6️⃣  Coefficients non nuls :")
    coef_series = pd.Series(model.coef_, index=X_processed.columns)
    non_zero = coef_series[coef_series != 0].sort_values(key=np.abs, ascending=False)
    print(non_zero.head(30))

    # Étape 7 : retour à l’échelle euros
    print("\n7️⃣  Transformation inverse (log → euros) et évaluation ...")
    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)

    rmse = np.sqrt(((y_target - y_pred) ** 2).mean())
    mae = np.mean(np.abs(y_target - y_pred))
    r2 = 1 - ((y_target - y_pred) ** 2).sum() / ((y_target - y_target.mean()) ** 2).sum()

    print(f"RMSE (train) : {rmse:,.2f} €")
    print(f"MAE  (train) : {mae:,.2f} €")
    print(f"R²   (train) : {r2:.4f}")

    print("\n✅ Figures enregistrées :")
    print(" - lasso_coef_path_cost_claims.png")
    print(" - lasso_cv_curve_cost_claims.png")


if __name__ == "__main__":
    main()
