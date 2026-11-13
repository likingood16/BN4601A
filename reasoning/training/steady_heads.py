"""Retrain the urgency and pathway heads on reworded complaints (+ children) and compare with the heads in the app
(13 Nov 2026; feedback A2, A3, B1). CPU only, after hpc/jobs/reword_embed.pbs.

    python steady_heads.py --check            # old heads only: reproduce the numbers in REPORT_NOTES 5t
    python steady_heads.py                    # train the new heads, compare, save as *_steady (the app is not touched)

Same held-out test patients for old and new (KTAS + NHAMCS test; children's test for the children's score), tested once.
New heads go into the app only if they do at least as well on those patients (rule agreed 12 Nov):
  urgency: AUROC on the original test cards not lower (0.005 rounding allowed) and sensitivity (head + floor) not lower;
  pathways: caught on the original test cards not lower (0.5 point allowed) and false flags per patient not higher.
"Steadier" = on the same test patients with the reworded complaints: smaller change in the urgency score (mean |change|,
points) and fewer urgent-flag flips; for pathways, the same caught / false-flag numbers on the reworded cards.
Saves outputs/metrics/steady_heads.json and models/heads/{urgent__base4bit__steady,pathways__base4bit_steady}.joblib.
"""
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702  (reasoning/src etc. importable)
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("src", "training", "checks")]
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HEADS_DIR, OUTPUTS, PROCESSED
from heads import bootstrap_auc, operating_point
from pathway_heads import NAMES, fit_flagger

SENS = 0.90                       # the app's cut-off: catch 9 in 10 urgent patients on KTAS validation
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03]
EMB = OUTPUTS / "embeddings"


def vectors(table, name):
    e = np.load(EMB / f"{table}__{name}.npz", allow_pickle=True)
    return dict(zip(map(str, e["ids"]), e["X"]))


def load(orig_emb):
    """All cards (original adults, children, reworded) with their vector, one table."""
    orig = pd.read_parquet(PROCESSED / "cards_pool1.parquet").assign(orig_id=lambda d: d["id"], style="orig")
    kids = pd.read_parquet(PROCESSED / "cards_children.parquet").assign(orig_id=lambda d: d["id"], style="orig")
    parts, V = [orig], vectors("cards_pool1", orig_emb)
    if (EMB / "cards_children__base4bit.npz").exists():
        parts.append(kids)
        V |= vectors("cards_children", "base4bit")
    if (EMB / "cards_reworded__base4bit.npz").exists():
        from reword_cards import bad
        rw = pd.read_parquet(PROCESSED / "cards_reworded.parquet")
        keep = (rw["style"] != "r1") & ~rw.apply(lambda r: bad(r["complaint_before"], r["complaint_after"]), axis=1)
        print(f"reworded cards: {len(rw)}, used {int(keep.sum())} (r1 first try and empty/chatty rows left out)", flush=True)
        parts.append(rw[keep])
        V |= vectors("cards_reworded", "base4bit")
    cards = pd.concat(parts, ignore_index=True)
    cards = cards[cards["id"].isin(V)].reset_index(drop=True)
    X = np.stack([V[i] for i in cards["id"]])
    cards["complaint"] = cards["card"].str.extract(r"PRESENTING COMPLAINT: (.*)", expand=False).str.replace(
        r" \(injury\)$", "", regex=True)
    return cards, X


# ---------- urgency ----------
def fit_urgency(cards, X, train_rows):
    y = cards["urgent"].to_numpy()
    va = ((cards["split"] == "val") & (cards["source"] == "ktas") & (cards["style"] == "orig")).to_numpy()
    best = None
    for C in C_GRID:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[train_rows], y[train_rows])
        auc = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, m)
    _, C, head = best
    p_va = head.predict_proba(X[va])[:, 1]
    return {"head": head, "C": C, "threshold": float(np.quantile(p_va[y[va] == 1], 1 - SENS))}


def child_cut(model, cards, X):
    """Cut-off for children: catch 9 in 10 urgent children's VALIDATION patients (25 urgent: rough)."""
    va = ((cards["source"] == "nhamcs_child") & (cards["split"] == "val") & (cards["style"] == "orig")).to_numpy()
    if not va.any():
        return None
    p = model["head"].predict_proba(X[va])[:, 1]
    return float(np.quantile(p[cards["urgent"].to_numpy()[va] == 1], 1 - SENS))


def urgency_report(model, cards, X, cut_child=None):
    p = model["head"].predict_proba(X)[:, 1]
    y = cards["urgent"].to_numpy()
    floor = (cards["floor_priority"].to_numpy() <= 2)
    out = {}
    test = (cards["split"] == "test").to_numpy()
    for name, src in [("adults", ["ktas", "nhamcs"]), ("ktas", ["ktas"]), ("nhamcs", ["nhamcs"]), ("children", ["nhamcs_child"])]:
        rows = test & cards["source"].isin(src).to_numpy() & (cards["style"] == "orig").to_numpy()
        if not rows.any():
            continue
        cut = cut_child if (name == "children" and cut_child is not None) else model["threshold"]
        r = {"n": int(rows.sum()), "urgent": int(y[rows].sum()), "auroc": round(float(roc_auc_score(y[rows], p[rows])), 4),
             "auroc_95ci": [round(v, 3) for v in bootstrap_auc(y[rows], p[rows], n=1000)], "cut_off": round(cut, 4),
             "head_plus_floor": {k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in operating_point(y[rows], (p[rows] >= cut) | floor[rows]).items()}}
        # reworded versions of the same test patients: AUROC on them, and how far each patient's score moves
        rw = test & cards["source"].isin(src).to_numpy() & (cards["style"] != "orig").to_numpy()
        if rw.any():
            p_orig = pd.Series(p[rows], index=cards.loc[rows, "orig_id"])
            moved = np.abs(p[rw] - p_orig.reindex(cards.loc[rw, "orig_id"]).to_numpy())
            flag_o = p_orig.reindex(cards.loc[rw, "orig_id"]).to_numpy() >= cut
            r["reworded"] = {"n": int(rw.sum()), "auroc": round(float(roc_auc_score(y[rw], p[rw])), 4),
                             "score_change_mean_points": round(float(moved.mean() * 100), 2),
                             "score_change_p90_points": round(float(np.quantile(moved, 0.9) * 100), 2),
                             "urgent_flag_flips": round(float(((p[rw] >= cut) != flag_o).mean()), 4),
                             "head_plus_floor": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in
                                                 operating_point(y[rw], (p[rw] >= cut) | floor[rw]).items()}}
        out[name] = r
    return out


# ---------- pathways ----------
def pathway_labels(cards):
    lab = pd.read_parquet(PROCESSED / "pathway_labels.parquet").set_index("id")[NAMES]
    have = cards["orig_id"].isin(lab.index).to_numpy()
    Y = np.zeros((len(cards), len(NAMES)), dtype=int)
    Y[have] = lab.loc[cards.loc[have, "orig_id"]].to_numpy()
    return Y, have


def pathway_report(fl, cards, X, Y, have):
    """caught = share of (patient, pathway it needed) pairs the app would show; false flags = pathways shown that the
    patient did not need, per patient. Uses PathwayFlagger.flags(), the exact rule the app shows (at most 3, lead)."""
    out = {}
    names = list(fl.heads)
    texts = {fl.heads[n]["flag"]: n for n in names}
    from pathway_heads import PATHWAYS
    texts |= {PATHWAYS[n][0]: n for n in names}
    for style in ("orig", "reworded"):
        rows = np.where(have & (cards["split"] == "test").to_numpy() & cards["source"].isin(["ktas", "nhamcs"]).to_numpy()
                        & ((cards["style"] == "orig") if style == "orig" else (cards["style"] != "orig")).to_numpy())[0]
        if not len(rows):
            continue
        shown = np.zeros((len(rows), len(names)), dtype=bool)
        for k, i in enumerate(rows):
            for t in fl.flags(X[i], cards.at[i, "complaint"]):
                shown[k, names.index(texts[t])] = True
        need = Y[rows].astype(bool)
        out[style] = {"n": int(len(rows)), "caught": round(float(shown[need].mean()), 4),
                      "false_flags_per_patient": round(float((shown & ~need).sum(1).mean()), 3),
                      "no_pathway_patients_flagged": round(float(shown[~need.any(1)].any(1).mean()), 4),
                      "per_pathway_caught": {n: round(float(shown[need[:, j], j].mean()), 3) for j, n in enumerate(names)
                                             if need[:, j].any()},
                      "per_pathway_false_flag_rate": {n: round(float(shown[~need[:, j], j].mean()), 4) for j, n in enumerate(names)}}
    return out


def main(check, orig_emb):
    cards, X = load(orig_emb)
    print(cards.groupby(["source", "style", "split"]).size().unstack(fill_value=0).to_string(), flush=True)
    old_u = joblib.load(HEADS_DIR / "urgent__base4bit__ktas_nhamcs.joblib")
    old_u = {"head": old_u["head"], "threshold": old_u["thresholds"][SENS]}
    old_p = joblib.load(HEADS_DIR / "pathways__base4bit.joblib")
    Y, have = pathway_labels(cards)
    res = {"orig_embeddings": orig_emb, "old": {"urgency": urgency_report(old_u, cards, X, child_cut(old_u, cards, X)),
                                                "pathways": pathway_report(old_p, cards, X, Y, have)}}
    print(json.dumps(res["old"], indent=1), flush=True)
    if check:
        return
    adult_train = (cards["split"] == "train").to_numpy() & cards["source"].isin(["ktas", "nhamcs"]).to_numpy()
    kid_train = (cards["split"] == "train").to_numpy() & (cards["source"] == "nhamcs_child").to_numpy()
    orig = (cards["style"] == "orig").to_numpy()
    variants = {
        "reworded": adult_train,                                   # adults, original + 2 rewordings each
        "reworded_children": adult_train | kid_train,              # + children (original + rewordings)
        "children_only_orig": adult_train & orig | kid_train & orig,   # children added, no rewording (to separate the two)
    }
    res["new"] = {}
    for name, rows in variants.items():
        m = fit_urgency(cards, X, rows)
        res["new"][name] = {"C": m["C"], "n_train": int(rows.sum()),
                            "urgency": urgency_report(m, cards, X, child_cut(m, cards, X))}
        print(name, json.dumps(res["new"][name], indent=1), flush=True)
        joblib.dump({"head": m["head"], "threshold": m["threshold"], "thresholds": {SENS: m["threshold"]},
                     "child_threshold": child_cut(m, cards, X), "embeddings": "base4bit", "trained_on": name},
                    HEADS_DIR / f"urgent__base4bit__steady_{name}.joblib")

    # pathways: same method (fit_flagger), training rows = adults' original + reworded cards; C and cut-offs on the
    # ORIGINAL validation cards, as before
    tr = adult_train & have
    va = (cards["split"] == "val").to_numpy() & orig & have
    fl = fit_flagger(X, cards["complaint"].fillna("").tolist(), Y, tr, va)
    res["new"]["pathways_reworded"] = pathway_report(fl, cards, X, Y, have)
    print("pathways_reworded", json.dumps(res["new"]["pathways_reworded"], indent=1), flush=True)
    joblib.dump(fl, HEADS_DIR / "pathways__base4bit_steady.joblib")
    (OUTPUTS / "metrics" / "steady_heads.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="old heads only")
    ap.add_argument("--orig-emb", default="base4bit", help="vectors of the original adult cards (laptop: base4bit)")
    a = ap.parse_args()
    main(a.check, a.orig_emb)
