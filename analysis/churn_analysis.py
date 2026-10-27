"""
RetentIA — analyse statistique et modélisation du churn (reproduction Python du pipeline Dataiku).

Usage :
    python analysis/churn_analysis.py data/Telco-Customer-Churn.csv

Étapes :
    1. Qualité des données
    2. Analyse bivariée : taux de churn avec IC de Wilson, test du chi², V de Cramér, test de Mann-Whitney
    3. Régression logistique interprétable (odds ratios et IC à 95 %)
    4. Comparaison de modèles en validation croisée stratifiée (ROC-AUC, PR-AUC, Brier)
    5. Évaluation sur un jeu de test indépendant et courbe de gains
Tous les résultats sont écrits dans results/metrics.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "results"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion k/n (plus fiable que Wald pour p proche de 0 ou 1)."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def cramers_v(table: pd.DataFrame) -> float:
    """V de Cramér : taille d'effet du chi², comprise entre 0 (indépendance) et 1 (liaison parfaite)."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges est lu comme texte : 11 valeurs vides correspondent à des clients d'ancienneté 0 (pas encore facturés).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.drop(columns=["customerID"])


def quality(df_raw: pd.DataFrame) -> dict:
    blank = (df_raw["TotalCharges"].astype(str).str.strip() == "").sum()
    return {
        "rows": int(len(df_raw)),
        "columns": int(df_raw.shape[1]),
        "duplicates_customerID": int(df_raw["customerID"].duplicated().sum()),
        "blank_TotalCharges": int(blank),
        "blank_TotalCharges_all_tenure_0": bool((df_raw.loc[df_raw["TotalCharges"].astype(str).str.strip() == "", "tenure"] == 0).all()),
    }


def bivariate(df: pd.DataFrame) -> dict:
    res = {"categorical": {}, "numeric": {}}
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in cats:
        table = pd.crosstab(df[c], df["Churn"])
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        levels = {}
        for lvl, row in table.iterrows():
            n, k = int(row.sum()), int(row.get(1, 0))
            lo, hi = wilson_ci(k, n)
            levels[str(lvl)] = {"n": n, "churn_rate": k / n, "ci95": [lo, hi]}
        res["categorical"][c] = {"chi2": chi2, "dof": int(dof), "p_value": p, "cramers_v": cramers_v(table), "levels": levels}
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        a, b = df.loc[df.Churn == 1, c], df.loc[df.Churn == 0, c]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rank_biserial = 1 - 2 * u / (len(a) * len(b))  # taille d'effet dans [-1, 1]
        res["numeric"][c] = {
            "median_churn": float(a.median()),
            "median_stay": float(b.median()),
            "mannwhitney_U": float(u),
            "p_value": float(p),
            "rank_biserial": float(rank_biserial),
        }
    return res


def tenure_buckets(df: pd.DataFrame) -> dict:
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72"]
    g = df.groupby(pd.cut(df["tenure"], bins=bins, labels=labels), observed=True)["Churn"].agg(["sum", "count"])
    return {str(k): {"n": int(r["count"]), "churn_rate": float(r["sum"] / r["count"])} for k, r in g.iterrows()}


def logit_odds_ratios(df: pd.DataFrame) -> dict:
    """Régression logistique sur un jeu de variables réduit et peu corrélé, pour une lecture causale-prudente des effets."""
    X = pd.DataFrame({
        "contract_one_year": (df.Contract == "One year").astype(int),
        "contract_two_year": (df.Contract == "Two year").astype(int),
        "internet_fiber": (df.InternetService == "Fiber optic").astype(int),
        "internet_none": (df.InternetService == "No").astype(int),
        "payment_electronic_check": (df.PaymentMethod == "Electronic check").astype(int),
        "paperless_billing": (df.PaperlessBilling == "Yes").astype(int),
        "tech_support": (df.TechSupport == "Yes").astype(int),
        "online_security": (df.OnlineSecurity == "Yes").astype(int),
        "senior": (df.SeniorCitizen == "Yes").astype(int),
        "tenure_per_year": df.tenure / 12.0,
        "monthly_charges_per_10": df.MonthlyCharges / 10.0,
    })
    model = sm.Logit(df["Churn"], sm.add_constant(X)).fit(disp=0)
    ci = model.conf_int()
    out = {}
    for name in X.columns:
        out[name] = {
            "odds_ratio": float(np.exp(model.params[name])),
            "ci95": [float(np.exp(ci.loc[name, 0])), float(np.exp(ci.loc[name, 1]))],
            "p_value": float(model.pvalues[name]),
        }
    return {"pseudo_r2_mcfadden": float(model.prsquared), "n": int(model.nobs), "coefficients": out}


def build_models(cat_cols, num_cols) -> dict:
    pre_linear = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    pre_tree = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    return {
        "Régression logistique": Pipeline([("pre", pre_linear), ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "Forêt aléatoire": Pipeline([("pre", pre_tree), ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1))]),
        "Gradient boosting": Pipeline([("pre", pre_tree), ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED))]),
    }


def gains(y_true: np.ndarray, score: np.ndarray, deciles=(0.1, 0.2, 0.3, 0.5)) -> dict:
    """Part des churners captés en ciblant les x % de clients les plus à risque, et lift associé."""
    order = np.argsort(-score)
    y_sorted = y_true[order]
    total = y_sorted.sum()
    out = {}
    for q in deciles:
        k = int(round(q * len(y_sorted)))
        captured = y_sorted[:k].sum() / total
        out[f"top_{int(q * 100)}pct"] = {"share_of_churners_captured": float(captured), "lift": float(captured / q), "precision": float(y_sorted[:k].mean())}
    return out


def rounded(obj, nd: int = 4):
    """Arrondit récursivement les flottants (p-values en notation scientifique à 3 chiffres significatifs)."""
    if isinstance(obj, dict):
        return {k: rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, nd) for v in obj]
    if isinstance(obj, float):
        return float(f"{obj:.3g}") if 0 < abs(obj) < 1e-3 else round(obj, nd)
    return obj


def main(path: str) -> None:
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(path)
    df = load(path)
    report = {"quality": quality(raw), "base_churn_rate": float(df.Churn.mean())}
    lo, hi = wilson_ci(int(df.Churn.sum()), len(df))
    report["base_churn_rate_ci95"] = [lo, hi]
    report["bivariate"] = bivariate(df)
    report["tenure_buckets"] = tenure_buckets(df)
    report["logit"] = logit_odds_ratios(df)

    y = df.pop("Churn").to_numpy()
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report["cv"], report["test"] = {}, {}
    for name, model in build_models(cat_cols, num_cols).items():
        s = cross_validate(model, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision", "neg_brier_score"])
        report["cv"][name] = {
            "roc_auc_mean": float(s["test_roc_auc"].mean()), "roc_auc_std": float(s["test_roc_auc"].std()),
            "pr_auc_mean": float(s["test_average_precision"].mean()), "pr_auc_std": float(s["test_average_precision"].std()),
            "brier_mean": float(-s["test_neg_brier_score"].mean()),
        }
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        report["test"][name] = {
            "roc_auc": float(roc_auc_score(y_test, p)),
            "pr_auc": float(average_precision_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "gains": gains(y_test, p),
        }
    report["test_size"] = int(len(y_test))
    report["test_churn_rate"] = float(y_test.mean())

    (OUT / "metrics.json").write_text(json.dumps(rounded(report), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(rounded({k: report[k] for k in ("cv", "test")}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/Telco-Customer-Churn.csv")
