"""ML predicts the triage LEVEL (1-5) from the MedGemma vector, with honest uncertainty (conformal prediction sets).

    python level_model.py --emb base4bit

1. Level model: multinomial logistic regression on the same vectors as the urgency head, trained on KTAS + NHAMCS
   train. C and class weighting are chosen by 5-fold cross-validation on TRAIN (quadratic weighted kappa), so the
   whole validation set stays free for step 2.
2. Conformal sets (split conformal, "LAC"): on validation, score = 1 - p(true level); the cut-off q is the
   ceil((n+1)(1-alpha))/n quantile. A new patient's set = every level with p >= 1 - q. Guarantee: the true level
   is in the set for at least 1-alpha of new patients (alpha = 0.10 -> 90%).
3. Test once on KTAS + NHAMCS test, plus the same 490 patients MedGemma was tested on (answers.py), so the ML
   level can be compared with the levels MedGemma wrote.
"Safe level" = ML level, but at most 2 when the urgency head + Caleb's floor flag the patient (never lower).
Saves models/heads/level__<emb>.joblib and outputs/metrics/level_model__<emb>.json.
"""
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702  (reasoning/src etc. importable)
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("src", "training", "checks")]
import argparse
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HEADS_DIR, OUTPUTS, PROCESSED, SEED
from heads import load_xy

LEVELS = np.array([1, 2, 3, 4, 5])
GRID = [(C, w) for C in (0.001, 0.003, 0.01) for w in (None, "balanced")]


def model(C, weight):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=weight, max_iter=3000))


def qwk(y, pred):
    """Quadratic weighted kappa: agreement on an ordered scale (1 = perfect, 0 = chance). Nurses reach ~0.77-0.83."""
    return float(cohen_kappa_score(y, pred, weights="quadratic"))


def level_scores(y, pred, sets=None):
    r = {"n": int(len(y)), "exact": round(float(np.mean(pred == y)), 4),
         "within_1": round(float(np.mean(np.abs(pred - y) <= 1)), 4), "weighted_kappa": round(qwk(y, pred), 4),
         "urgent_sensitivity": round(float(np.mean(pred[y <= 2] <= 2)), 4),
         "urgent_specificity": round(float(np.mean(pred[y > 2] > 2)), 4),
         "predicted_level_counts": {int(k): int((pred == k).sum()) for k in LEVELS}}
    if sets is not None:
        sizes = sets.sum(1)
        r["conformal"] = {"coverage": round(float(sets[np.arange(len(y)), y - 1].mean()), 4),
                          "mean_set_size": round(float(sizes.mean()), 3),
                          "single_level": round(float(np.mean(sizes == 1)), 4),
                          "uncertain_2plus": round(float(np.mean(sizes >= 2)), 4),
                          "urgent_patients_set_includes_1or2": round(float(sets[y <= 2][:, :2].any(1).mean()), 4)}
    return r


def run(emb, alpha):
    cards, X = load_xy(emb)
    cards = cards.merge(pd.read_parquet(PROCESSED / "head_scores.parquet"), on="id", how="left")
    y = cards["triage_level"].to_numpy().astype(int)
    tr = ((cards["split"] == "train") & cards["source"].isin(["ktas", "nhamcs"])).to_numpy()
    va, te = (cards["split"] == "val").to_numpy(), (cards["split"] == "test").to_numpy()

    # 1. choose C + class weighting by cross-validation on TRAIN only
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = {}
    for C, w in GRID:
        p = cross_val_predict(model(C, w), X[tr], y[tr], cv=folds, method="predict_proba")
        cv[(C, w)] = qwk(y[tr], LEVELS[p.argmax(1)])
        print(f"  C={C} weight={w}: CV weighted kappa {cv[(C, w)]:.3f}", flush=True)
    C, w = max(cv, key=cv.get)
    m = model(C, w).fit(X[tr], y[tr])
    P = m.predict_proba(X)                          # columns = levels 1..5

    # 2. conformal cut-off from VALIDATION patients (never used for fitting)
    n = int(va.sum())
    scores = 1 - P[va, y[va] - 1]
    q = float(np.quantile(scores, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
    sets = P >= 1 - q

    # 3. test once
    pred = LEVELS[P.argmax(1)]
    safe = np.where(cards["flagged"].fillna(False).to_numpy(), np.minimum(pred, 2), pred)
    test_ktas, test_nh = te & (cards["source"] == "ktas").to_numpy(), te & (cards["source"] == "nhamcs").to_numpy()
    # the same 490 patients answers.py used (all KTAS test + 300 NHAMCS test, seed 42)
    nh_ids = cards[test_nh].sample(300, random_state=SEED)["id"]
    same490 = test_ktas | cards["id"].isin(nh_ids).to_numpy()
    res = {"embeddings": emb, "chosen": {"C": C, "class_weight": w}, "cv_weighted_kappa": {f"{k[0]}|{k[1]}": round(v, 4)
                                                                                          for k, v in cv.items()},
           "alpha": alpha, "conformal_cutoff": q}
    for name, mask in (("test_ktas", test_ktas), ("test_nhamcs", test_nh), ("test_same490", same490)):
        res[name] = {"ml_level": level_scores(y[mask], pred[mask], sets[mask]),
                     "safe_level": level_scores(y[mask], safe[mask])}
    joblib.dump({"model": m, "conformal_cutoff": q, "alpha": alpha, "embeddings": emb}, HEADS_DIR / f"level__{emb}.joblib")
    out = OUTPUTS / "metrics" / f"level_model__{emb}.json"
    out.write_text(json.dumps(res, indent=1))
    return res, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="base4bit")
    ap.add_argument("--alpha", type=float, default=0.10, help="0.10 = 90% guaranteed coverage")
    a = ap.parse_args()
    res, path = run(a.emb, a.alpha)
    print("chosen:", res["chosen"], "| conformal cut-off:", round(res["conformal_cutoff"], 3))
    for name in ("test_ktas", "test_nhamcs", "test_same490"):
        for kind in ("ml_level", "safe_level"):
            print(name, kind, json.dumps(res[name][kind]))
    print("saved", path)
