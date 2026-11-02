"""Experiment 1: MedGemma vector + structured vitals in one urgency head, with the extra evaluation.

    python fusion.py --emb base4bit

Variants (all trained on KTAS+NHAMCS train; settings chosen on KTAS val; test used once):
  text     = MedGemma vector only (same as heads.py)
  vitals   = structured features only (age, sex, arrival, injury, pain, vitals, NEWS2 + missing flags)
  concat   = vector + structured features side by side in one logistic regression
  stacked  = out-of-fold text-head score + structured features in a small second logistic regression
Caleb's vitals_p_urgent is NOT used as a feature: his dev-refit model saw KTAS val, which would leak into model choice.

Extra evaluation per test set: paired bootstrap of the AUROC gain over `text`, calibration (Brier, ECE),
operating points at 80% and 90% sensitivity targets (+ Caleb's floor), subgroups (age 65+, sex), decision curve.
Saves outputs/metrics/fusion__<emb>.json.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import OUTPUTS, PROCESSED, SEED
from heads import C_GRID, load_xy, operating_point

NUM = ["age", "nrs_pain", "sbp", "dbp", "hr", "rr", "temp_c", "spo2", "news2"]
FLAGS = ["ambulance", "injured"]


def structured(cards, train_mask):
    """Numeric table: medians from TRAIN fill the gaps, plus a 0/1 'was missing' column for each."""
    p = pd.read_parquet(PROCESSED / "pool1.parquet").set_index("id").loc[cards["id"]].reset_index()
    p["news2"] = cards["news2"].to_numpy()
    out = pd.DataFrame({"male": (p["sex"].str.lower().str.startswith("m")).astype(float)})
    for c in NUM + FLAGS:
        v = p[c].astype(float)
        out[c] = v.fillna(v[train_mask].median())
        out[c + "_missing"] = v.isna().astype(float)
    return out.loc[:, out[train_mask].std() > 0].to_numpy()


def fit_lr(X, y, tr, va):
    """Pick C on KTAS val by AUROC, return (fitted model, best C, val AUROC)."""
    best = None
    for C in C_GRID:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[tr], y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        if best is None or auc > best[2]:
            best = (m, C, auc)
    return best


def ece(y, p, bins=10):
    """Expected calibration error: average gap between predicted and real urgent rate, over 10 bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def paired_gain(y, p_new, p_old, n=2000):
    """Bootstrap the SAME patients for both models: 95% CI of the AUROC gain and how often it is <= 0."""
    rng, d = np.random.default_rng(SEED), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
    d = np.array(d)
    return {"gain": float(roc_auc_score(y, p_new) - roc_auc_score(y, p_old)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_no_gain": float((d <= 0).mean())}


def net_benefit(y, p, t):
    """Decision curve (Vickers & Elkin 2006): true alarms minus false alarms weighted by the threshold odds."""
    flag = p >= t
    return float(((flag & (y == 1)).sum() - (flag & (y == 0)).sum() * t / (1 - t)) / len(y))


def run(emb):
    cards, E = load_xy(emb)
    y = cards["urgent"].to_numpy()
    tr = ((cards["split"] == "train") & cards["source"].isin(["ktas", "nhamcs"])).to_numpy()
    va = ((cards["split"] == "val") & (cards["source"] == "ktas")).to_numpy()
    S = structured(cards, tr)

    text, C_text, _ = fit_lr(E, y, tr, va)
    # stacking needs text-head scores on TRAIN that the head did not fit on: 5-fold out-of-fold predictions
    oof = np.zeros(len(y))
    oof[tr] = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=C_text, max_iter=5000)),
                                E[tr], y[tr], cv=5, method="predict_proba")[:, 1]
    oof[~tr] = text.predict_proba(E[~tr])[:, 1]
    logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1))

    feats = {"text": E, "vitals": S, "concat": np.hstack([E, S]), "stacked": np.column_stack([logit, S])}
    models, p, results = {}, {}, {"embeddings": emb, "variants": {}}
    for name, X in feats.items():
        m, C, val_auc = (text, C_text, roc_auc_score(y[va], text.predict_proba(E[va])[:, 1])) if name == "text" \
            else fit_lr(X, y, tr, va)
        p[name] = m.predict_proba(X)[:, 1]
        results["variants"][name] = {"best_C": C, "val_auroc": round(val_auc, 4)}
    best = max(results["variants"], key=lambda k: results["variants"][k]["val_auroc"])
    results["chosen_on_val"] = best

    floor = cards["floor_priority"].to_numpy() <= 2
    age = pd.read_parquet(PROCESSED / "pool1.parquet").set_index("id").loc[cards["id"], "age"].to_numpy()
    sex = pd.read_parquet(PROCESSED / "pool1.parquet").set_index("id").loc[cards["id"], "sex"].str.lower().str[0].to_numpy()
    for src in ["ktas", "nhamcs"]:
        te = ((cards["split"] == "test") & (cards["source"] == src)).to_numpy()
        yt, r = y[te], {}
        for name in feats:
            pt = p[name][te]
            r[name] = {"auroc": round(float(roc_auc_score(yt, pt)), 4), "brier": round(float(brier_score_loss(yt, pt)), 4),
                       "ece": round(ece(yt, pt), 4)}
            if name != "text":
                r[name]["vs_text"] = paired_gain(yt, pt, p["text"][te])
            for target in (0.80, 0.90):
                thr = float(np.quantile(p[name][va][y[va] == 1], 1 - target))
                r[name][f"at_{int(target * 100)}"] = operating_point(yt, (pt >= thr) | floor[te])
        pb = p[best][te]
        r["subgroups_" + best] = {g: round(float(roc_auc_score(yt[m], pb[m])), 4) for g, m in
                                  {"age_65_plus": age[te] >= 65, "age_under_65": age[te] < 65,
                                   "male": sex[te] == "m", "female": sex[te] == "f"}.items() if len(set(yt[m])) == 2}
        r["decision_curve"] = {str(t): {"treat_all": net_benefit(yt, np.ones(len(yt)), t),
                                        "text": net_benefit(yt, p["text"][te], t), best: net_benefit(yt, pb, t)}
                               for t in (0.05, 0.1, 0.2, 0.3)}
        results[f"test_{src}"] = r

    out = OUTPUTS / "metrics" / f"fusion__{emb}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    return results, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="base4bit")
    res, path = run(ap.parse_args().emb)
    print("val AUROC:", {k: v["val_auroc"] for k, v in res["variants"].items()}, "-> chosen:", res["chosen_on_val"])
    for src in ["ktas", "nhamcs"]:
        r = res[f"test_{src}"]
        print(f"\n== test {src}")
        for k in ["text", "vitals", "concat", "stacked"]:
            v, g = r[k], r[k].get("vs_text")
            gain = f" | gain vs text {g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] p={g['p_no_gain']:.3f}" if g else ""
            a80, a90 = v["at_80"], v["at_90"]
            print(f"{k:8s} AUROC {v['auroc']:.3f} Brier {v['brier']:.3f} ECE {v['ece']:.3f}{gain}")
            print(f"         80%: sens {a80['sensitivity']:.2f} spec {a80['specificity']:.2f} flagged {a80['flagged']} | "
                  f"90%: sens {a90['sensitivity']:.2f} spec {a90['specificity']:.2f} flagged {a90['flagged']}")
        print("subgroups:", {k: v for k, v in r.items() if k.startswith("subgroups")})
        print("decision curve (net benefit):", r["decision_curve"])
    print("saved", path)
