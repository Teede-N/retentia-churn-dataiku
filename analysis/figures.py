"""
RetentIA — génération des figures du README (dossier figures/).

Usage :
    python analysis/figures.py data/Telco-Customer-Churn.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn_analysis import SEED, build_models, load, logit_odds_ratios, wilson_ci  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def dec(v, _=None):
    return f"{v:.1f}".replace(".", ",")


def pct(x):
    return f"{x * 100:.1f} %".replace(".", ",")


def fig_contract_tenure(df):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    order = [("Month-to-month", "Mensuel"), ("One year", "1 an"), ("Two year", "2 ans")]
    rates, lo, hi = [], [], []
    for key, _ in order:
        s = df.loc[df.Contract == key, "Churn"]
        r = s.mean()
        l, h = wilson_ci(int(s.sum()), len(s))
        rates.append(r), lo.append(r - l), hi.append(h - r)
    x = np.arange(3)
    a.bar(x, rates, width=0.55, color=BLUE, yerr=[lo, hi], capsize=5, error_kw={"ecolor": INK2, "lw": 1.2})
    for i, r in enumerate(rates):
        a.text(i, r + hi[i] + 0.012, pct(r), ha="center", color=INK, fontsize=11)
    a.set_xticks(x, [lab for _, lab in order])
    a.set_ylim(0, 0.55)
    a.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    a.set_title("Taux de churn selon le contrat")
    a.set_ylabel("Taux de churn (IC 95 % Wilson)")
    a.grid(axis="x", visible=False)

    bins, labels = [-1, 6, 12, 24, 48, 72], ["0-6", "7-12", "13-24", "25-48", "49-72"]
    g = df.groupby(pd.cut(df.tenure, bins=bins, labels=labels), observed=True)["Churn"].mean()
    b.plot(range(5), g.values, color=ORANGE, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for i, r in enumerate(g.values):
        b.text(i, r + 0.03, pct(r), ha="center", color=INK, fontsize=10)
    b.set_xticks(range(5), labels)
    b.set_xlabel("Ancienneté (mois)")
    b.set_ylim(0, 0.65)
    b.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    b.set_title("Taux de churn selon l'ancienneté")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "01_contrat_anciennete.png", dpi=150)
    plt.close(fig)


def fig_odds_ratios(df):
    names = {
        "contract_two_year": "Contrat 2 ans", "contract_one_year": "Contrat 1 an", "internet_none": "Sans internet",
        "online_security": "Sécurité en ligne", "tech_support": "Support technique", "tenure_per_year": "Ancienneté (+1 an)",
        "monthly_charges_per_10": "Montant mensuel (+10 $)", "senior": "Client senior", "paperless_billing": "Facture dématérialisée",
        "payment_electronic_check": "Chèque électronique", "internet_fiber": "Fibre optique",
    }
    coefs = logit_odds_ratios(df)["coefficients"]
    items = sorted(coefs.items(), key=lambda kv: kv[1]["odds_ratio"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, (k, v) in enumerate(items):
        c = BLUE if v["odds_ratio"] < 1 else ORANGE
        ax.plot(v["ci95"], [i, i], color=c, lw=2)
        ax.plot(v["odds_ratio"], i, "o", color=c, ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(v["ci95"][1] * 1.06, i, f"{v['odds_ratio']:.2f}".replace(".", ","), va="center", color=INK, fontsize=10)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 3], ["0,2", "0,5", "1", "2", "3"])
    ax.set_yticks(range(len(items)), [names[k] for k, _ in items])
    ax.set_xlabel("Odds ratio (échelle log, IC 95 %)   ← protège du churn | aggrave le churn →")
    ax.set_title("Régression logistique : odds ratios ajustés")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.15, 3.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "02_odds_ratios.png", dpi=150)
    plt.close(fig)


def fig_models(df):
    y = df.pop("Churn").to_numpy()
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    probas = {}
    for (name, model), color in zip(build_models(cat_cols, ["tenure", "MonthlyCharges", "TotalCharges"]).items(), [BLUE, ORANGE, AQUA]):
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        probas[name] = p
        fpr, tpr, _ = roc_curve(y_te, p)
        a.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc_score(y_te, p):.3f})".replace("0.", "0,"))
    a.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Hasard (AUC = 0,5)")
    a.xaxis.set_major_formatter(dec)
    a.yaxis.set_major_formatter(dec)
    a.set_xlabel("Taux de faux positifs")
    a.set_ylabel("Taux de vrais positifs")
    a.set_title("Courbes ROC (jeu de test)")
    a.legend(frameon=False, fontsize=9, loc="lower right")

    p = probas["Régression logistique"]
    order = np.argsort(-p)
    captured = np.concatenate([[0], np.cumsum(y_te[order]) / y_te.sum()])
    share = np.linspace(0, 1, len(captured))
    b.plot(share, captured, color=BLUE, lw=2, label="Régression logistique")
    b.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Ciblage au hasard")
    k = int(round(0.2 * len(y_te)))
    b.plot(share[k], captured[k], "o", color=BLUE, ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    b.annotate(f"20 % des clients → {pct(captured[k])} des départs", (share[k], captured[k]), xytext=(0.36, 0.22),
               color=INK, fontsize=10, arrowprops={"arrowstyle": "->", "color": INK2})
    b.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    b.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    b.set_xlabel("Part des clients ciblés (du plus au moins risqué)")
    b.set_ylabel("Part des départs captés")
    b.set_title("Courbe de gains")
    b.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "03_roc_gains.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Calibration parfaite")
    ax.plot(mean_pred, frac_pos, color=BLUE, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=2, label="Régression logistique")
    ax.xaxis.set_major_formatter(dec)
    ax.yaxis.set_major_formatter(dec)
    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Fréquence observée de churn")
    ax.set_title("Calibration des probabilités (déciles)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "04_calibration.png", dpi=150)
    plt.close(fig)


def main(path):
    FIG.mkdir(exist_ok=True)
    df = load(path)
    fig_contract_tenure(df)
    fig_odds_ratios(df)
    fig_models(df.copy())
    print("Figures écrites dans", FIG)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/Telco-Customer-Churn.csv")
