"""Urgency score for EVERY Pool 1 patient, as the app would show it to MedGemma.

    python head_scores.py --emb base4bit

Uses the saved head (models/heads/urgent__<emb>__ktas_nhamcs.joblib) and its 80%-sensitivity threshold.
Training patients get OUT-OF-FOLD scores (5 folds: each patient scored by a head that never saw them), so
training examples built from them are as realistic as validation/test patients.
Flag = head says urgent OR Caleb's floor says priority 1-2.
Saves data/processed/head_scores.parquet: id, p_urgent, head_flag, floor_flag, flagged.
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from config import HEADS_DIR, PROCESSED, SEED
from heads import load_xy


def run(emb):
    saved = joblib.load(HEADS_DIR / f"urgent__{emb}__ktas_nhamcs.joblib")
    cards, X = load_xy(emb)
    y = cards["urgent"].to_numpy()
    tr = ((cards["split"] == "train") & cards["source"].isin(["ktas", "nhamcs"])).to_numpy()
    p = saved["head"].predict_proba(X)[:, 1]
    p[tr] = cross_val_predict(clone(saved["head"]), X[tr], y[tr], cv=5, method="predict_proba")[:, 1]
    out = pd.DataFrame({"id": cards["id"], "p_urgent": p, "head_flag": p >= saved["threshold"],
                        "floor_flag": cards["floor_priority"].to_numpy() <= 2})
    out["flagged"] = out["head_flag"] | out["floor_flag"]
    out.to_parquet(PROCESSED / "head_scores.parquet", index=False)
    for split in ("train", "test"):
        m = (cards["split"] == split).to_numpy()
        print(f"{split}: flagged {out['flagged'][m].mean():.0%}, urgent caught {out['flagged'][m & (y == 1)].mean():.0%}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="base4bit")
    run(ap.parse_args().emb)
