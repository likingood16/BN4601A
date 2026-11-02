"""Is the urgency score's test result luck? And would more data help?

    python robustness.py --emb base4bit --splits 20

1. Repeated splits: re-split ALL Pool 1 patients N times (70/15/15, same urgent share per source each time).
   Each time: choose C on validation, set the 80% / 90% alarm thresholds on validation, test once.
   The MedGemma head and a vitals-only model (stand-in for Caleb's, refit each time) see the SAME splits.
2. Learning curve: on the original seed-42 split, train the head on 10-100% of the training patients
   (3 random draws each) and test. Still rising at 100% = more patients would help; flat = they would not.
The original seed-42 test result stays the headline; this shows how much it moves.
Saves outputs/metrics/robustness__<emb>.json.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import OUTPUTS, SEED
from fusion import structured
from heads import load_xy, operating_point

C_GRID = [0.001, 0.003, 0.01, 0.03]          # around the C the full search picked (0.003)


def head(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_on_val(X, y, tr, va):
    scores = {C: roc_auc_score(y[va], head(C).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]) for C in C_GRID}
    return head(max(scores, key=scores.get)).fit(X[tr], y[tr])


def one_split(seed, cards, X, S, y):
    idx = np.arange(len(y))
    strata = cards["source"].astype(str) + "_" + cards["urgent"].astype(str)
    tr_i, rest = train_test_split(idx, test_size=0.30, stratify=strata, random_state=seed)
    va_i, te_i = train_test_split(rest, test_size=0.50, stratify=strata.iloc[rest], random_state=seed)
    tr, va, te = (np.isin(idx, i) for i in (tr_i, va_i, te_i))
    floor = cards["floor_priority"].to_numpy() <= 2
    out = {}
    for name, F in (("medgemma", X), ("vitals_only", S)):
        p = fit_on_val(F, y, tr, va).predict_proba(F)[:, 1]
        r = {}
        for src in ("ktas", "nhamcs"):
            m = te & (cards["source"] == src).to_numpy()
            r[f"auroc_{src}"] = float(roc_auc_score(y[m], p[m]))
        for target in (0.80, 0.90):
            thr = np.quantile(p[va][y[va] == 1], 1 - target)
            o = operating_point(y[te], (p[te] >= thr) | floor[te])
            r[f"at_{int(target * 100)}"] = {k: o[k] for k in ("sensitivity", "specificity", "ppv", "flagged", "n")}
            r[f"at_{int(target * 100)}"]["missed_urgent"] = int(round((1 - o["sensitivity"]) * y[te].sum()))
        out[name] = r
    return out


def summary(values):
    v = np.array(values)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


def learning_curve(cards, X, y):
    tr = ((cards["split"] == "train")).to_numpy()
    va = ((cards["split"] == "val") & (cards["source"] == "ktas")).to_numpy()
    C = max(C_GRID, key=lambda c: roc_auc_score(y[va], head(c).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]))
    rng, tr_idx, curve = np.random.default_rng(SEED), np.flatnonzero(tr), {}
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        res = {"ktas": [], "nhamcs": []}
        for _ in range(1 if frac == 1.0 else 3):
            pick = np.zeros(len(y), bool)
            pick[rng.choice(tr_idx, int(frac * len(tr_idx)), replace=False)] = True
            p = head(C).fit(X[pick], y[pick]).predict_proba(X)[:, 1]
            for src in res:
                m = ((cards["split"] == "test") & (cards["source"] == src)).to_numpy()
                res[src].append(roc_auc_score(y[m], p[m]))
        curve[str(frac)] = {"n_train": int(frac * len(tr_idx)), **{s: round(float(np.mean(v)), 4) for s, v in res.items()}}
        print("  learning curve", frac, curve[str(frac)], flush=True)
    return {"C": C, "curve": curve}


def run(emb, n_splits):
    cards, X = load_xy(emb)
    y = cards["urgent"].to_numpy()
    S = structured(cards, (cards["split"] == "train").to_numpy())
    runs = []
    for seed in range(n_splits):
        runs.append(one_split(seed, cards, X, S, y))
        r = runs[-1]
        print(f"split {seed:2d}: MedGemma KTAS {r['medgemma']['auroc_ktas']:.3f} NHAMCS {r['medgemma']['auroc_nhamcs']:.3f} | "
              f"vitals-only KTAS {r['vitals_only']['auroc_ktas']:.3f} NHAMCS {r['vitals_only']['auroc_nhamcs']:.3f}", flush=True)
    res = {"embeddings": emb, "n_splits": n_splits, "splits": runs}
    for name in ("medgemma", "vitals_only"):
        res[name] = {k: summary([r[name][k] for r in runs]) for k in ("auroc_ktas", "auroc_nhamcs")}
        for t in ("at_80", "at_90"):
            res[name][t] = {k: summary([r[name][t][k] for r in runs]) for k in ("sensitivity", "specificity", "missed_urgent", "flagged")}
    res["medgemma_wins"] = {src: sum(r["medgemma"][f"auroc_{src}"] > r["vitals_only"][f"auroc_{src}"] for r in runs)
                            for src in ("ktas", "nhamcs")}
    res["learning_curve"] = learning_curve(cards, X, y)
    out = OUTPUTS / "metrics" / f"robustness__{emb}.json"
    out.write_text(json.dumps(res, indent=1))
    return res, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="base4bit")
    ap.add_argument("--splits", type=int, default=20)
    a = ap.parse_args()
    res, path = run(a.emb, a.splits)
    for name in ("medgemma", "vitals_only"):
        r = res[name]
        print(f"\n{name}: KTAS AUROC {r['auroc_ktas']} | NHAMCS AUROC {r['auroc_nhamcs']}")
        for t in ("at_80", "at_90"):
            print(f"  {t}: sens {r[t]['sensitivity']['mean']:.2f} (sd {r[t]['sensitivity']['sd']:.2f}) "
                  f"spec {r[t]['specificity']['mean']:.2f} missed {r[t]['missed_urgent']['mean']:.1f} of the test urgent")
    print("\nMedGemma beats vitals-only in", res["medgemma_wins"], "of", a.splits, "splits")
    print("saved", path)
