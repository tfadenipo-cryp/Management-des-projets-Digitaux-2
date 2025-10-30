#Choix des vars pertinentes 
#Var Cible : y = Cost_claims_year (C’est le coût total des sinistres supporté par l’assureur pour ce client sur l’année.)

# Lasso (équivalent glmnet) pour prédire Cost_claims_year

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.metrics import mean_squared_error
from scipy import sparse

# ========= CONFIG =========
# 
CSV_PATH = Path(r"C:\Users\hp\Desktop\Cours M2 SEP\management-2\management-des-projets-digitaux-2\data\raw\new_motor_vehicle_insurance_data.csv")
TARGET = "cost_claims_year"
CV_FOLDS = 10
RANDOM_STATE = 42

# ========= LOAD =========
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
assert TARGET in df.columns, f"Cible '{TARGET}' introuvable. Colonnes: {df.columns.tolist()}"
df = df.loc[~df[TARGET].isna()].copy()

X_df = df.drop(columns=[TARGET], errors="ignore")
y    = df[TARGET].values

# ========= MATRICE DE DESIGN (XX) =========
# - Numériques -> imputés médiane
# - Catégorielles -> OneHotEncoder en SPARSE (évite 10+ Go de RAM)
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_df.columns if c not in num_cols]

num_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
    # pas de StandardScaler ici: on standardise TOUT d'un coup juste après pour mimer scale(XX)
])

cat_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))  # <--- sparse !
])

prep = ColumnTransformer([
    ("num", num_tf, num_cols),
    ("cat", cat_tf, cat_cols)
], remainder="drop", sparse_threshold=1.0)  # force le sparse si possible

# On obtient XX en sparse
XX = prep.fit_transform(X_df)                   # scipy.sparse matrix
# ========= SCALE(XX) (centrage-réduction) =========
# Pour une matrice creuse, on ne centre pas (with_mean=False), mais on **met à l’échelle** (écart-type)
scaler = StandardScaler(with_mean=False)        # important: garder le sparse
XX_scaled = scaler.fit_transform(XX)            # toujours sparse

# (Option) convertir en CSC (plus rapide pour lasso_path)
XX_scaled = XX_scaled.tocsc()

# ========= CHEMIN DES COEFFICIENTS (comme plot(glmnet)) =========
# lasso_path calcule les coefs pour une grille d'alphas (lambdas)
# NOTE: lasso_path accepte le sparse CSC
alphas, coefs, _ = lasso_path(XX_scaled, y, alphas=None, max_iter=5000, verbose=False)

plt.figure(figsize=(8,6))
for i in range(coefs.shape[0]):
    plt.plot(np.log10(alphas), coefs[i, :])
plt.xlabel("log10(lambda)")
plt.ylabel("Coefficient")
plt.title("Chemin des coefficients — LASSO (équiv. glmnet)")
plt.tight_layout()
plt.savefig("lasso_coef_path.png", bbox_inches="tight")
plt.close()

# ========= CV LASSO pour lambda.min (équiv. cv.glmnet) =========
# LassoCV accepte aussi le sparse; on lui donne la même matrice standardisée
lasso_cv = LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE, alphas=100, max_iter=5000, n_jobs=-1)
lasso_cv.fit(XX_scaled, y)

lambda_min = lasso_cv.alpha_
mse_path_mean = lasso_cv.mse_path_.mean(axis=1)   # moyenne des MSE sur les folds
best_idx = list(lasso_cv.alphas_).index(lambda_min)
best_mse = float(mse_path_mean[best_idx])
best_rmse = float(np.sqrt(best_mse))

# Courbe de CV (comme plot(cv.glmnet))
plt.figure(figsize=(8,6))
plt.plot(np.log10(lasso_cv.alphas_), mse_path_mean, marker="o", ms=3)
plt.axvline(np.log10(lambda_min), linestyle="--", label=f"lambda.min={lambda_min:.4g}")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE (CV)")
plt.title("Validation croisée — LASSO")
plt.legend()
plt.tight_layout()
plt.savefig("lasso_cv_curve.png", bbox_inches="tight")
plt.close()

# ========= COEFS AU LAMBDA.MIN (équiv. coef(reg.cvlasso)) =========
# Récupérer les noms de colonnes finales de XX (num + dummies)
feature_names = []
feature_names.extend(num_cols)
if cat_cols:
    ohe = prep.named_transformers_["cat"].named_steps["ohe"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names.extend(ohe_names)

coef_series = pd.Series(lasso_cv.coef_, index=feature_names)
non_zero = coef_series[coef_series != 0].sort_values(key=np.abs, ascending=False)

# ========= SORTIES =========
print("\n=== LASSO (équivalent glmnet) — Résumé ===")
print(f"lambda.min (alpha*) : {lambda_min:.6g}")
print(f"CV MSE (min)        : {best_mse:.6g}")
print(f"CV RMSE (min)       : {best_rmse:.6g}")
print(f"Nb de features non nulles : {(coef_series!=0).sum()} / {len(feature_names)}")

print("\nTop 30 coefficients non nuls (|coef| décroissant) :")
print(non_zero.head(30))

print("\nFichiers générés : lasso_coef_path.png  |  lasso_cv_curve.png")
