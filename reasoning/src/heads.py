"""Score heads: logistic regression on MedGemma vectors -> p(urgent), then Caleb's safety floor.

    python heads.py --emb base                      # train on KTAS train, tune C on KTAS val, test once
    python heads.py --emb base --train-on ktas+nhamcs

Protocol (no peeking): C is chosen on the validation split; the alert threshold is set on validation to
catch 80% of urgent patients (Caleb's rule); the test split is used once, at the end.
Final decision = head flags urgent OR Caleb's floor says priority 1-2 (the floor can only raise urgency).
Saves models/heads/<name>.joblib and outputs/metrics/<name>.json.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HEADS_DIR, OUTPUTS, PROCESSED

C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
TARGET_SENS = 0.80


def load_xy(emb_name, table="cards_pool1"):
    cards = pd.read_parquet(PROCESSED / f"{table}.parquet")
    e = np.load(OUTPUTS / "embeddings" / f"{table}__{emb_name}.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(e["ids"])}
    cards = cards[cards["id"].isin(pos)].reset_index(drop=True)
    X = e["X"][[pos[i] for i in cards["id"]]]
    return cards, X


def operating_point(y, flag):
    y, flag = np.asarray(y).astype(bool), np.asarray(flag).astype(bool)
    tp, fn = (flag & y).sum(), (~flag & y).sum()
    fp, tn = (flag & ~y).sum(), (~flag & ~y).sum()
    div = lambda a, b: float(a / b) if b else float("nan")
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp),
            "npv": div(tn, tn + fn), "under_triage": div(fn, tp + fn), "over_triage": div(fp, fp + tn),
            "flagged": int(flag.sum()), "n": int(len(y))}


def bootstrap_auc(y, p, n=2000, seed=0):
    rng, y, p = np.random.default_rng(seed), np.asarray(y), np.asarray(p)
    stats = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
             if y[i].min() != y[i].max()]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate(cards, p, threshold):
    y = cards["urgent"].to_numpy()
    head_flag = p >= threshold
    final_flag = head_flag | (cards["floor_priority"].to_numpy() <= 2)
    lo, hi = bootstrap_auc(y, p)
    return {"auroc": float(roc_auc_score(y, p)), "auroc_95ci": [lo, hi],
            "brier": float(brier_score_loss(y, p)),
            "spearman_vs_level": float(spearmanr(p, -cards["triage_level"]).statistic),
            "head_only": operating_point(y, head_flag), "head_plus_floor": operating_point(y, final_flag)}


def run(emb, train_on="ktas", name=None):
    cards, X = load_xy(emb)
    sources = train_on.split("+")
    tr = (cards["split"] == "train") & cards["source"].isin(sources)
    va = (cards["split"] == "val") & (cards["source"] == "ktas")
    y = cards["urgent"].to_numpy()

    # 1. choose C on KTAS validation
    scores = {}
    for C in C_GRID:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[tr], y[tr])
        scores[C] = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
    best_C = max(scores, key=scores.get)
    head = make_pipeline(StandardScaler(), LogisticRegression(C=best_C, max_iter=5000)).fit(X[tr], y[tr])

    # 2. threshold on validation: flag 80% of urgent patients
    p_va = head.predict_proba(X[va])[:, 1]
    threshold = float(np.quantile(p_va[y[va] == 1], 1 - TARGET_SENS))

    # 3. test once: KTAS test (compare with Caleb) and NHAMCS test (other country)
    p = head.predict_proba(X)[:, 1]
    results = {"embeddings": emb, "train_on": train_on, "n_train": int(tr.sum()), "best_C": best_C,
               "val_auroc_by_C": {str(k): round(v, 4) for k, v in scores.items()}, "threshold": threshold}
    for src in ["ktas", "nhamcs"]:
        te = ((cards["split"] == "test") & (cards["source"] == src)).to_numpy()
        if te.any():
            results[f"test_{src}"] = evaluate(cards[te].reset_index(drop=True), p[te], threshold)
            results[f"test_{src}"]["vitals_model_auroc"] = float(
                roc_auc_score(y[te], cards.loc[te, "vitals_p_urgent"]))   # Caleb's model on the same patients

    name = name or f"urgent__{emb}__{train_on.replace('+', '_')}"
    HEADS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "metrics").mkdir(parents=True, exist_ok=True)
    joblib.dump({"head": head, "threshold": threshold, "embeddings": emb}, HEADS_DIR / f"{name}.joblib")
    (OUTPUTS / "metrics" / f"{name}.json").write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="base")
    ap.add_argument("--train-on", default="ktas", choices=["ktas", "ktas+nhamcs", "nhamcs"])
    a = ap.parse_args()
    r = run(a.emb, a.train_on)
    print(json.dumps({k: v for k, v in r.items() if k != "val_auroc_by_C"}, indent=1))
