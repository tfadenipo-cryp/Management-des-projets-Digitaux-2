#Target variable (y) : cost_claims_year
#Elle mesure le coût annuel total des sinistres (en euros) supporté par la compagnie 
# d’assurance pour chaque assuré.
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
g

if __name__ == "__main__":
    main()
############################

# ========= ÉVALUATION COMPLÈTE DU MODÈLE =========
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import TweedieRegressor
import numpy as np
import pandas as pd

y = y_target.to_numpy() if hasattr(y_target, "to_numpy") else np.asarray(y_target)

# 1) Statistiques de la cible (contexte métier)
print("\n--- Stats de la variable cible (cost_claims_year) ---")
print(f"n          : {y.size:,}")
print(f"moyenne    : {y.mean():,.2f}")
print(f"écart-type : {y.std():,.2f}")
print(f"médiane    : {np.median(y):,.2f}")
print(f"min / max  : {y.min():,.2f} / {y.max():,.2f}")
pct_zero = (y == 0).mean() * 100
print(f"% de zéros : {pct_zero:.2f}%")

# 2) Baselines simples
y_mean = np.full_like(y, y.mean())
rmse_mean = mean_squared_error(y, y_mean, squared=False)
mae_mean  = mean_absolute_error(y, y_mean)
print("\n--- Baseline (prédire la moyenne) ---")
print(f"RMSE baseline : {rmse_mean:,.2f}")
print(f"MAE  baseline : {mae_mean:,.2f}")

# 3) R² (train) et R² ajusté (train) du LASSO au lambda.min
r2_train = model.score(X_scaled, y)  # = r2_score(y, model.predict(X_scaled))
p = int(np.count_nonzero(model.coef_))
n = y.shape[0]
r2_adj_train = 1 - (1 - r2_train) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
print("\n--- R² sur apprentissage ---")
print(f"R² (train)        : {r2_train:.4f}")
print(f"R² ajusté (train) : {r2_adj_train:.4f} (p={p}, n={n})")

# 4) Scores en validation croisée (plus honnêtes)
print("\n--- Scores en validation croisée (CV) pour le LASSO ---")
lasso_final = Lasso(alpha=model.alpha_, max_iter=5000)

# R² CV
r2_cv = cross_val_score(lasso_final, X_scaled, y, cv=10, scoring="r2", n_jobs=-1).mean()
print(f"R² (CV 10-fold)   : {r2_cv:.4f}")

# RMSE CV
neg_mse_cv = cross_val_score(lasso_final, X_scaled, y, cv=10,
                             scoring="neg_mean_squared_error", n_jobs=-1)
rmse_cv = np.sqrt(-neg_mse_cv.mean())
print(f"RMSE (CV 10-fold) : {rmse_cv:,.2f}   |  ratio vs moyenne = {rmse_cv / y.mean():.3f}")

# MAE CV
neg_mae_cv = cross_val_score(lasso_final, X_scaled, y, cv=10,
                             scoring="neg_mean_absolute_error", n_jobs=-1)
mae_cv = -neg_mae_cv.mean()
print(f"MAE  (CV 10-fold) : {mae_cv:,.2f}")

# RMSLE CV (utile si cible très asymétrique, zéros gérés par log1p)
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(np.clip(y_true, 0, None)))**2))

# scorer custom (positif -> on inverse le signe pour cross_val_score)
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import make_scorer

rmsle_scorer = make_scorer(lambda yt, yp: -rmsle(yt, yp))  # négatif pour compatibilité

neg_rmsle_cv = cross_val_score(lasso_final, X_scaled, y, cv=10,
                               scoring=rmsle_scorer, n_jobs=-1)
rmsle_cv = -neg_rmsle_cv.mean()
print(f"RMSLE (CV 10-fold): {rmsle_cv:.4f}")

# 5) Comparaison avec un GLM Tweedie (souvent pertinent en assurance)
print("\n--- Comparaison : GLM Tweedie (power≈1.5, lien log) ---")
tweedie = TweedieRegressor(power=1.5, alpha=1.0, link="log", max_iter=1000)
r2_cv_tw = cross_val_score(tweedie, X_scaled, y, cv=10, scoring="r2", n_jobs=-1).mean()
neg_mse_tw = cross_val_score(tweedie, X_scaled, y, cv=10,
                             scoring="neg_mean_squared_error", n_jobs=-1)
rmse_cv_tw = np.sqrt(-neg_mse_tw.mean())
print(f"Tweedie R² (CV)   : {r2_cv_tw:.4f}")
print(f"Tweedie RMSE (CV) : {rmse_cv_tw:,.2f}")

# 6) Récap tableau
summary = pd.DataFrame({
    "Modèle": ["Baseline (mean)", "LASSO (CV)", "Tweedie (CV)"],
    "R²": [np.nan, r2_cv, r2_cv_tw],
    "RMSE": [rmse_mean, rmse_cv, rmse_cv_tw],
    "MAE": [mae_mean, mae_cv, np.nan],
    "RMSE / mean(y)": [rmse_mean / y.mean(), rmse_cv / y.mean(), rmse_cv_tw / y.mean()]
})
print("\n=== RÉCAP ===")
print(summary.to_string(index=False))
