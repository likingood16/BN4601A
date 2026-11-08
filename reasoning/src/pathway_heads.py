"""ED pathway flags by AI (8 Nov 2026): one small trained head per pathway on MedGemma's vector of the patient card,
replacing the word rules in ed_assessment.PATHWAYS (kept on the git branch rules-reasoning).

    python pathway_heads.py              # labels -> train on train, choose C and cut-off on val, test once
    python pathway_heads.py --labels     # only rebuild the labels table

Answer key ("did this patient need this pathway's work-up?"): the FINAL diagnosis is in the pathway's group
(NHAMCS ICD-10 codes DIAG1-5; KTAS diagnosis names), or the coded/written complaint names it. The diagnosis is only
the answer key, never an input: the head sees the same card as the urgency head (no diagnosis on it).
So the head can flag "looks like a stroke" when the words don't say stroke, which the word rules cannot.

Model (try 3, 8 Nov): a gate head "needs any pathway?" + one head per pathway, all logistic regression on MedGemma's
vector (PCA 256) + the complaint's words (TF-IDF), C chosen on validation. Cut-offs on validation: gate and each head
catch 90%. Try 1-2 (7 separate heads on the vector only) flagged 64% of patients who needed no pathway.
Test: KTAS + NHAMCS test splits, once. Compared with the word rules on the same test patients.
Saves models/heads/pathways__base4bit.joblib and outputs/metrics/pathways__base4bit.json.
"""
import argparse
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HEADS_DIR, OUTPUTS, PROCESSED, REPO

sys.path.insert(0, str(REPO / "vitals" / "src"))
EMB = "base4bit"
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
TARGET_SENS = 0.90
GATE_SENS = 0.90          # the gate lets through 90% of patients who needed some pathway (val)

# pathway: (what the app tells the clinician, ICD-10 prefixes, words in a KTAS diagnosis, words in the complaint)
PATHWAYS = {
    "chest pain": ("chest pain -> ECG within 10 minutes of arrival",
                   ("I20", "I21", "I22", "I23", "I24", "I25", "R07"),
                   r"angina|coronary|myocardial infarct|\bnstemi\b|\bstemi\b|\bmi\b|chest pain",
                   r"chest (?:pain|tight|discomfort|pressure)|angina"),
    "possible stroke": ("possible stroke -> record time last known well, glucose, urgent brain imaging",
                        ("I60", "I61", "I62", "I63", "I64", "G45", "G46", "R47"),
                        r"stroke|cerebral infarct|cerebral h(?:a)?emorrh|intracerebral|transient ischemic|\btia\b|hemipleg|dysarthri|aphasi",
                        r"stroke|weakness of one side|hemipleg|slurred|speech|facial droop|\bcva\b"),
    "breathing": ("breathing complaint -> oxygen saturation and work of breathing",
                  ("J12", "J13", "J14", "J15", "J16", "J17", "J18", "J20", "J21", "J4", "J8", "J9", "R06", "I50", "I26"),
                  r"pneumonia|asthma|copd|chronic obstructive|dyspn|respiratory failure|pulmonary (?:edema|embol)|heart failure|bronchi",
                  r"dyspn|short(?:ness)? of breath|breath|wheez|respirat"),
    "collapse/syncope": ("collapse/syncope -> ECG, glucose, check for injury",
                         ("R55", "R40", "I44", "I45", "I46", "I47", "I49", "E16"),
                         r"syncop|faint|collapse|loss of consciousness|arrhythm|bradycard|hypoglyc",
                         r"syncop|faint|collapse|loss of consciousness|passed out"),
    "seizure": ("seizure -> glucose, airway protection, time of seizure",
                ("R56", "G40", "G41"), r"seizure|convuls|epilep|status epilepticus", r"seiz|convuls|epilep|fit\b"),
    "possible poisoning": ("possible poisoning -> substance, amount and time not recorded",
                           tuple(f"T{n}" for n in range(36, 66)) + ("F100", "F110", "F120", "F130", "F140", "F150", "F160", "F190"),
                           r"poison|intoxic|overdose|toxic effect|drug ingestion",
                           r"overdose|poison|intoxic|ingest"),
    "possible infection": ("possible infection -> sepsis screen (qSOFA)",
                           # infections that can become sepsis, the screen's purpose. 8 Nov: "any A/B code" (viral
                           # illness, gastroenteritis, ...) made the head flag 45% of patients needing no pathway.
                           ("A40", "A41", "A49", "R652", "R7881", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69",
                            "N10", "N390", "N159", "L02", "L03", "G00", "G03", "G04", "K35", "K57", "K65", "K81", "K83",
                            "M00", "M86", "I33", "R50"),
                           r"sepsis|septic|infect|pneumonia|cellulitis|pyelonephr|urinary tract infection|fever|abscess|meningitis|cholecystitis",
                           r"fever|infect|sepsis|chills|rigors"),
}
NAMES = list(PATHWAYS)


def labels():
    """id -> 0/1 per pathway, from final diagnoses (+ coded complaint). Matches pool1 ids (ktas_<visit_id>,
    nhamcs_<row after Caleb's harmonize>)."""
    import ktas_data as kd
    import nhamcs
    k = kd.clean(kd.load_raw())
    k = k[0] if isinstance(k, tuple) else k
    kt = pd.DataFrame({"id": "ktas_" + k["visit_id"].astype(str), "dx": k["diagnosis"].fillna("").str.lower(),
                       "cc": k["chief_complaint"].fillna("").str.lower(), "codes": ""})
    raw = nhamcs.load()
    diag = pd.read_stata(nhamcs.RAW, columns=["DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5"], convert_categoricals=False)
    raw = raw.join(diag[[c for c in diag if c not in raw]])
    kept = raw[(raw.AGE >= 16) & raw.IMMEDR.between(1, 5)].reset_index(drop=True)   # same rows as harmonize()
    codes = kept[["DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5"]].astype(str).apply(
        lambda r: " ".join(c.rstrip("-") for c in r if c not in ("-9", "-8", "-7", "nan", "")), axis=1)
    pool = pd.read_parquet(PROCESSED / "pool1.parquet")
    nh_cc = pool[pool.source == "nhamcs"].set_index("id")["chief_complaint"].fillna("").str.lower()
    nh = pd.DataFrame({"id": [f"nhamcs_{i}" for i in range(len(kept))], "dx": "", "codes": codes.to_numpy()})
    nh["cc"] = nh["id"].map(nh_cc).fillna("")
    df = pd.concat([kt, nh], ignore_index=True)
    for name, (_, icd, dx_words, cc_words) in PATHWAYS.items():
        by_code = df["codes"].apply(lambda s: any(c.startswith(icd) for c in s.split()))
        df[name] = (by_code | df["dx"].str.contains(dx_words) | df["cc"].str.contains(cc_words)).astype(int)
        df[f"{name}__hidden"] = ((by_code | df["dx"].str.contains(dx_words)) & ~df["cc"].str.contains(cc_words)).astype(int)
    df = df.merge(pool[["id", "split", "source", "chief_complaint"]], on="id", how="inner")
    df.to_parquet(PROCESSED / "pathway_labels.parquet", index=False)
    print(df.groupby("split")[NAMES].mean().round(3).to_string(), "\nn:", df.groupby("split").size().to_dict(), flush=True)
    return df


def rules_flag(complaint):
    """The old app's word rules (ed_assessment.PATHWAYS, same 7 pathways in the same order; kept on branch
    rules-reasoning), for comparison. Not the label words above, which would grade the rules with their own key."""
    from ed_assessment import PATHWAYS as OLD
    text = complaint if isinstance(complaint, str) else ""
    return {n: bool(re.search(w, text, re.I)) for n, (w, _) in zip(NAMES, OLD)}


class PathwayFlagger:
    """Gate + 7 heads (structure test 8 Nov, outputs/metrics/pathway_structures.json): best of 7 layouts.
    Input = MedGemma's vector of the card (PCA 256) + the complaint words (TF-IDF), both learned from data.
    The gate ("needs any pathway work-up?") must say yes before any pathway head may flag: halved false flags
    (64% -> 31% of no-pathway patients) for a small loss in catch (0.86 -> 0.81)."""

    def __init__(self, scaler, pca, pca_sd, tfidf, gate, gate_cut, heads):
        self.scaler, self.pca, self.pca_sd, self.tfidf = scaler, pca, pca_sd, tfidf
        self.gate, self.gate_cut, self.heads = gate, gate_cut, heads

    def features(self, vectors, complaints):
        import scipy.sparse as sps
        v = self.pca.transform(self.scaler.transform(np.asarray(vectors).reshape(len(complaints), -1))) / self.pca_sd
        words = self.tfidf.transform([c.lower() if isinstance(c, str) else "" for c in complaints])
        return sps.hstack([sps.csr_matrix(v) * 0.1, words]).tocsr()

    def scores(self, vectors, complaints):
        F = self.features(vectors, complaints)
        return self.gate.predict_proba(F)[:, 1], np.column_stack([h["head"].predict_proba(F)[:, 1] for h in self.heads.values()])

    def flag_matrix(self, vectors, complaints):
        g, P = self.scores(vectors, complaints)
        return (P >= np.array([h["cut_off"] for h in self.heads.values()])) & (g >= self.gate_cut)[:, None]

    def flags(self, vector, complaint):
        """One patient -> the flag texts the app shows."""
        row = self.flag_matrix(np.asarray(vector).reshape(1, -1), [complaint])[0]
        return [h["flag"] for h, on in zip(self.heads.values(), row) if on]


def _fit(F, y, tr, va, Cs=(0.01, 0.1, 1)):
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(F[tr], y[tr])
        auc = roc_auc_score(y[va], m.predict_proba(F[va])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, m)
    return best


def fit_flagger(X, cc, Y, tr, va):
    """Fit the gate + 7 heads on rows `tr`; C and every cut-off chosen on rows `va`. Used by main() and by
    learning.py (same method on the original patients + the patients seen in use)."""
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pathway_heads as me          # saved as pathway_heads.PathwayFlagger, so the app can load it
    scaler = StandardScaler().fit(X[tr])
    pca = PCA(256, random_state=0).fit(scaler.transform(X[tr]))
    pca_sd = pca.transform(scaler.transform(X[tr])).std()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([c.lower() for i, c in enumerate(cc) if tr[i]])
    fl = me.PathwayFlagger(scaler, pca, pca_sd, tfidf, None, None, {})
    F = fl.features(X, cc)
    any_y = Y.max(1)
    _, gC, gate = _fit(F, any_y, tr, va)
    gv = gate.predict_proba(F[va])[:, 1]
    heads = {}
    for j, name in enumerate(NAMES):
        _, C, m = _fit(F, Y[:, j], tr, va)
        pv = m.predict_proba(F[va])[:, 1]
        heads[name] = {"head": m, "cut_off": float(np.quantile(pv[Y[va, j] == 1], 1 - TARGET_SENS)), "flag": PATHWAYS[name][0], "C": C}
    fl.gate, fl.gate_cut, fl.heads = gate, float(np.quantile(gv[any_y[va] == 1], 1 - GATE_SENS)), heads
    fl.gate_C = gC
    return fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    lab = labels()
    if a.labels:
        return
    e = np.load(OUTPUTS / "embeddings" / f"cards_pool1__{EMB}.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(e["ids"])}
    lab = lab[lab["id"].isin(pos)].reset_index(drop=True)
    X = e["X"][[pos[i] for i in lab["id"]]]
    cc = lab["chief_complaint"].fillna("").astype(str).tolist()
    sp = {s: (lab["split"] == s).to_numpy() for s in ("train", "val", "test")}
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Y = lab[NAMES].to_numpy()
    any_y = Y.max(1)
    fl = fit_flagger(X, cc, Y, tr, va)
    res = {"emb": EMB, "structure": "gate + 7 heads on vector(PCA256) + complaint words", "gate_C": fl.gate_C,
           "gate_catch_on_val": GATE_SENS, "target_sensitivity_on_val": TARGET_SENS,
           "n": {s: int(m.sum()) for s, m in sp.items()}, "pathways": {}}
    heads = fl.heads
    flags = fl.flag_matrix(X[te], [c for i, c in enumerate(cc) if te[i]])
    _, Pt = fl.scores(X[te], [c for i, c in enumerate(cc) if te[i]])
    rule = pd.DataFrame([rules_flag(c) for c in lab["chief_complaint"]]).to_numpy()[te]
    none = any_y[te] == 0
    for j, name in enumerate(NAMES):
        yt, ft, rt = Y[te, j] == 1, flags[:, j], rule[:, j]
        hidden = lab[f"{name}__hidden"].to_numpy()[te] == 1
        res["pathways"][name] = {
            "C": heads[name]["C"], "cut_off": round(heads[name]["cut_off"], 4), "test_with": int(yt.sum()),
            "test_auroc": round(float(roc_auc_score(yt, Pt[:, j])), 3),
            "ai_sensitivity": round(float(ft[yt].mean()), 3), "ai_specificity": round(float((~ft[~yt]).mean()), 3),
            "rules_sensitivity": round(float(rt[yt].mean()), 3), "rules_specificity": round(float((~rt[~yt]).mean()), 3),
            "hidden_test": int(hidden.sum()),
            "ai_caught_hidden": round(float(ft[hidden].mean()), 3) if hidden.any() else None,
            "rules_caught_hidden": round(float(rt[hidden].mean()), 3) if hidden.any() else None}
        print(name, res["pathways"][name], flush=True)
    res["no_pathway_patients_flagged"] = {"ai": round(float(flags[none].any(1).mean()), 3),
                                          "rules": round(float(rule[none].any(1).mean()), 3)}
    res["flags_per_patient"] = {"ai": round(float(flags.sum(1).mean()), 2), "rules": round(float(rule.sum(1).mean()), 2)}
    print(res["no_pathway_patients_flagged"], res["flags_per_patient"], flush=True)
    HEADS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fl, HEADS_DIR / f"pathways__{EMB}.joblib")
    (OUTPUTS / "metrics" / f"pathways__{EMB}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
