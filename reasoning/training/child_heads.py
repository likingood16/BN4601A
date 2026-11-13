"""Children's urgency score (under 16, NHAMCS 2022): is a children's model better than the adult one? (13 Nov 2026, A2)

    python child_heads.py --quick     # no MedGemma: age + vital signs + complaint words, CPU, seconds
    python child_heads.py             # + MedGemma vectors (needs outputs/embeddings/cards_children__base4bit.npz)

Data: training/child_data.py (1,795 children, 170 urgent; 70/15/15 split, seed 42). Protocol as for the adults:
C chosen on the children's validation split, cut-off = catch 9 in 10 urgent children on validation (only 25 urgent
there, so the cut-off is rough), test once on the 270 children's test patients (26 urgent).
Compared on the same 270 children: Caleb's vitals model (adult, as in the app), the app's MedGemma head (adult) and the
children's models. Saves outputs/metrics/child_heads.json; the MedGemma children's head goes to
models/heads/urgent__base4bit__children.joblib (not used by the app until it is shown to be better).
"""
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702  (reasoning/src etc. importable)
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("src", "training", "checks")]
import argparse
import json

import joblib  # our own saved heads (trusted files made by this project)
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HEADS_DIR, OUTPUTS, PROCESSED
from heads import bootstrap_auc, operating_point

SENS = 0.90
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
NUM = ["age", "hr", "rr", "sbp", "dbp", "temp_c", "spo2", "nrs_pain", "ambulance", "injured"]


def report(y, p, cut, floor):
    lo, hi = bootstrap_auc(y, p, n=1000)
    rnd = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
    return {"auroc": round(float(roc_auc_score(y, p)), 3), "auroc_95ci": [round(lo, 3), round(hi, 3)],
            "cut_off": round(float(cut), 4), "score_only": rnd(operating_point(y, p >= cut)),
            "score_plus_floor": rnd(operating_point(y, (p >= cut) | floor))}


def fit(F, y, tr, va, balanced=False):
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced" if balanced else None).fit(F[tr], y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(F[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, m)
    _, C, m = best
    p = m.predict_proba(F)[:, 1]
    return m, C, p, float(np.quantile(p[va][y[va] == 1], 1 - SENS))


def structured(t, tr):
    """Age, vital signs (missing -> training median + a 'missing' flag) and the complaint's words."""
    num = t[NUM].astype(float)
    miss = num.isna().astype(float).add_suffix("_missing")
    num = num.fillna(num[tr].median())
    sc = StandardScaler().fit(num[tr])
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(t.loc[tr, "chief_complaint"].fillna(""))
    return sps.hstack([sps.csr_matrix(sc.transform(num)), sps.csr_matrix(miss.to_numpy()),
                       words.transform(t["chief_complaint"].fillna(""))]).tocsr()


def main(quick):
    t = pd.read_parquet(PROCESSED / "pool1_children.parquet")
    cards = pd.read_parquet(PROCESSED / "cards_children.parquet").set_index("id").loc[t["id"]].reset_index()
    y = t["urgent"].to_numpy()
    tr, va, te = [(t["split"] == s).to_numpy() for s in ("train", "val", "test")]
    floor = (cards["floor_priority"].to_numpy() <= 2)
    res = {"n": {s: int(m.sum()) for s, m in zip(("train", "val", "test"), (tr, va, te))},
           "urgent": {s: int(y[m].sum()) for s, m in zip(("train", "val", "test"), (tr, va, te))},
           "floor_alone_on_test": operating_point(y[te], floor[te])}

    # Caleb's vitals model (adult) as the app runs it: its own probability, cut-off set on the children's validation
    pv = cards["vitals_p_urgent"].to_numpy()
    res["vitals_model_adult"] = report(y[te], pv[te], np.quantile(pv[va][y[va] == 1], 1 - SENS), floor[te])

    F = structured(t, tr)
    m, C, p, cut = fit(F, y, tr, va, balanced=True)
    res["children_age_vitals_words"] = {"C": C, **report(y[te], p[te], cut, floor[te])}

    emb = OUTPUTS / "embeddings" / "cards_children__base4bit.npz"
    if not quick and emb.exists():
        e = np.load(emb, allow_pickle=True)
        pos = {i: k for k, i in enumerate(map(str, e["ids"]))}
        X = e["X"][[pos[i] for i in t["id"]]]
        app = joblib.load(HEADS_DIR / "urgent__base4bit__ktas_nhamcs.joblib")
        pa = app["head"].predict_proba(X)[:, 1]
        res["app_medgemma_head_adult_cutoff"] = report(y[te], pa[te], app["thresholds"][SENS], floor[te])
        res["app_medgemma_head_child_cutoff"] = report(y[te], pa[te], np.quantile(pa[va][y[va] == 1], 1 - SENS), floor[te])
        head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        best = None
        for Cc in C_GRID[:5]:
            h = make_pipeline(StandardScaler(), LogisticRegression(C=Cc, max_iter=5000)).fit(X[tr], y[tr])
            auc = roc_auc_score(y[va], h.predict_proba(X[va])[:, 1])
            if best is None or auc > best[0]:
                best = (auc, Cc, h)
        _, Cc, head = best
        ph = head.predict_proba(X)[:, 1]
        cut_h = float(np.quantile(ph[va][y[va] == 1], 1 - SENS))
        res["children_medgemma_head"] = {"C": Cc, **report(y[te], ph[te], cut_h, floor[te])}
        joblib.dump({"head": head, "threshold": cut_h, "thresholds": {SENS: cut_h}, "embeddings": "base4bit",
                     "trained_on": "NHAMCS 2022 children (under 16), train split"},
                    HEADS_DIR / "urgent__base4bit__children.joblib")
    (OUTPUTS / "metrics" / ("child_heads_quick.json" if quick else "child_heads.json")).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict) and "auroc" in v:
            print(f"{k:34s} AUROC {v['auroc']} {v['auroc_95ci']}  +floor: sens {v['score_plus_floor']['sensitivity']} "
                  f"spec {v['score_plus_floor']['specificity']} flagged {v['score_plus_floor']['flagged']}/{v['score_plus_floor']['n']}")
    print("n", res["n"], "urgent", res["urgent"], "floor alone", {k: res["floor_alone_on_test"][k] for k in ("sensitivity", "specificity", "flagged")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
