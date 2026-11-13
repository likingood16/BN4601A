"""Learning from use (8 Nov 2026): the app records what clinicians correct; an update step retrains on it, tests the new
models on the SAME frozen test patients as before, and switches only if they are not worse on safety.

    # in the app: nothing to do; corrections are logged as people work (web/case_files/learning/)
    python learning.py --status                  # what has been collected, and how the AI is doing in real use
    python learning.py --update                  # retrain + safety gate; switches models only if the gate passes
    python learning.py --update --dry-run        # same, but never switch (a report only)
    python learning.py --rollback                # put the previous models back

Why not learn live, inside the app (design choice, see REPORT_NOTES 5m):
  - a model that changes during a shift can't be checked; every version here is tested before use and kept;
  - feedback loops: a nurse ticking an AI to-do is not proof it was needed. Labels come from the clinician's FINAL
    diagnosis and level when the patient leaves, never from "accepted the AI's suggestion";
  - this is the regulators' model for changing clinical AI (FDA predetermined change control plan, Dec 2024): what may
    change, how it is tested, and what must not get worse are fixed in advance (GATE below).

What is logged (one JSON line per event, `feedback.jsonl`; vectors in `vectors/<patient>.npy`):
  numbers_corrected  the note(s), what the AI read, what the nurse corrected       -> number-reader training data
  todo               an AI to-do ticked done / removed, with its source             -> monitoring only (see above)
  outcome            when a patient is cleared: arrival fields, complaint, the AI's urgency/level/pathway flags,
                     where the patient went, and the clinician's final diagnosis and level (optional)
                                                                                     -> new labelled patients
What the update retrains (CPU, minutes): the urgency head and the pathway flagger, on the original training
patients + 80% of the patients seen in use (the other 20% are kept to report how the new models do on real use).
Cut-offs are still chosen on the original validation patients, and the gate uses the original test patients.
"""
import argparse
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NUMBER_FIELDS = ["age", "sex", "hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain", "mental", "ambulance", "injured"]
# The safety gate, fixed in advance. A candidate is used only if every line holds on the frozen test patients.
GATE = {"urgency_auroc_drop_max": 0.01,             # KTAS and NHAMCS test, each
        "urgency_sensitivity_drop_max": 0.0,        # urgent patients flagged (score + floor) must not fall
        "pathway_caught_drop_max": 0.02,            # mean over the 7 pathways
        "pathway_false_flag_rise_max": 0.02}        # share of no-pathway patients with any flag
MIN_NEW_PATIENTS = 20                               # fewer outcome cases than this: nothing to learn yet


class FeedbackLog:
    """Append-only log next to the board file. Thread-safe; one line per event."""

    def __init__(self, folder):
        self.folder = Path(folder)
        (self.folder / "vectors").mkdir(parents=True, exist_ok=True)
        self.file = self.folder / "feedback.jsonl"
        self._lock = threading.Lock()

    def add(self, kind, patient, **data):
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "patient": patient, **data}
        with self._lock, open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_plain) + "\n")
        return row

    def save_vector(self, patient, vector):
        if vector is not None:
            np.save(self.folder / "vectors" / f"{patient}.npy", np.asarray(vector, dtype=np.float32))

    def events(self, kind=None):
        if not self.file.exists():
            return []
        rows = [json.loads(line) for line in self.file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]

    def summary(self):
        """How the AI is doing in real use, from the corrections (shown in the app and by --status)."""
        nums, todos, outs = self.events("numbers_corrected"), self.events("todo"), self.events("outcome")
        per_field = {}
        for e in nums:
            for k in e.get("changed", []):
                per_field[k] = per_field.get(k, 0) + 1
        by_source = {}
        for t in todos:
            s = by_source.setdefault(t.get("by", "?"), {"done": 0, "removed": 0})
            if t["action"] in s:
                s[t["action"]] += 1
        with_vec = sum((self.folder / "vectors" / f"{o['patient']}.npy").exists() for o in outs)
        labelled = [o for o in outs if o.get("diagnosis") or o.get("final_level")]
        updates = _read_updates()
        return {"patients_with_outcome": len(outs), "with_diagnosis_or_level": len(labelled),
                "usable_for_learning": sum((self.folder / "vectors" / f"{o['patient']}.npy").exists()
                                           for o in labelled),
                "with_vector": with_vec, "needed_before_update": MIN_NEW_PATIENTS,
                "number_corrections": len(nums), "numbers_corrected_by_field": per_field,
                "ai_todos": by_source,
                "level_changed_by_clinician": sum(1 for o in outs if o.get("final_level") and o.get("ai_level")
                                                  and int(o["final_level"]) != int(o["ai_level"])),
                "last_update": updates[-1] if updates else None, "models_version": _version()}


def _plain(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---------------------------------------------------------------- the update step (offline)
def _paths():
    from config import HEADS_DIR, OUTPUTS
    return HEADS_DIR, OUTPUTS / "learning"


LIVE = {"urgent": "urgent__base4bit__ktas_nhamcs.joblib", "pathways": "pathways__base4bit.joblib"}


def _read_updates():
    _, out = _paths()
    f = out / "updates.json"
    return json.loads(f.read_text()) if f.exists() else []


def _version():
    """v1 = as trained; +1 per switch, -1 per rollback."""
    n, last = 1, " (as trained)"
    for u in _read_updates():
        if u.get("switched"):
            n, last = n + 1, f" (updated {u['time']})"
        elif str(u.get("result", "")).startswith("rolled back"):
            n, last = max(1, n - 1), f" (rolled back {u['time']})"
    return f"v{n}{last}"


def _in_use_split(pid):
    """Deterministic 80/20: the same patient always lands on the same side, so later updates never test on a patient
    an earlier update trained on."""
    return "train" if int(hashlib.sha1(pid.encode()).hexdigest(), 16) % 5 else "check"


def new_patients(log):
    """Outcome cases with a vector and a label -> table rows like Pool 1 (urgent from the clinician's final level,
    pathways from the final diagnosis words + the complaint, the same answer key as pathway_heads.labels)."""
    import re
    from pathway_heads import NAMES, PATHWAYS
    rows, vecs = [], []
    for o in log.events("outcome"):
        vf = log.folder / "vectors" / f"{o['patient']}.npy"
        if not vf.exists() or not (o.get("diagnosis") or o.get("final_level")):
            continue
        dx, cc = (o.get("diagnosis") or "").lower(), (o.get("complaint") or "").lower()
        f = o.get("fields", {})
        row = {"id": f"use_{o['patient']}", "split": "use_" + _in_use_split(o["patient"]), "chief_complaint": cc,
               "urgent": (int(o["final_level"]) <= 2) if o.get("final_level") else None,
               "floor_priority": o.get("floor_priority", 9), "has_dx": bool(dx)}
        for n in NAMES:
            _, _, dx_words, cc_words = PATHWAYS[n]
            row[n] = int(bool(re.search(dx_words, dx)) or bool(re.search(cc_words, cc))) if dx else None
        rows.append(row)
        vecs.append(np.load(vf))
    return rows, (np.stack(vecs) if vecs else None)


def _urgency_metrics(head, threshold, cards, X):
    from sklearn.metrics import roc_auc_score
    out = {}
    for src in ("ktas", "nhamcs"):
        te = ((cards["split"] == "test") & (cards["source"] == src)).to_numpy()
        y, p = cards.loc[te, "urgent"].to_numpy(), head.predict_proba(X[te])[:, 1]
        flag = (p >= threshold) | (cards.loc[te, "floor_priority"].to_numpy() <= 2)
        out[src] = {"auroc": round(float(roc_auc_score(y, p)), 4), "sensitivity": round(float(flag[y == 1].mean()), 4)}
    return out


def _pathway_metrics(fl, X, cc, Y):
    flags = fl.flag_matrix(X, cc)
    caught = [float(flags[Y[:, j] == 1, j].mean()) for j in range(Y.shape[1]) if (Y[:, j] == 1).any()]
    none = Y.max(1) == 0
    return {"caught_mean": round(float(np.mean(caught)), 4), "false_flag": round(float(flags[none].any(1).mean()), 4)}


def update(feedback, dry_run=False):
    """Retrain on original + in-use patients, run the gate, switch only if it passes. Returns the report."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import pathway_heads as ph
    from config import PROCESSED
    from heads import C_GRID, load_xy
    heads_dir, out_dir = _paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = FeedbackLog(feedback)
    rows, Xn = new_patients(log)
    report = {"time": time.strftime("%Y-%m-%d %H:%M"), "new_patients": len(rows), "gate": GATE, "switched": False}
    if len(rows) < MIN_NEW_PATIENTS:
        report["result"] = f"only {len(rows)} labelled patients from use (need {MIN_NEW_PATIENTS}); nothing changed"
        return _finish(report, out_dir)
    new = pd.DataFrame(rows)
    use_tr = (new["split"] == "use_train").to_numpy()

    # --- urgency head: same method as heads.run (C on KTAS val, cut-off catching 90% of urgent KTAS val patients)
    cards, X = load_xy("base4bit")
    current = joblib.load(heads_dir / LIVE["urgent"])
    cur_thr = current["thresholds"][0.9]
    nu = new[use_tr & new["urgent"].notna().to_numpy()]
    Xu = Xn[nu.index.to_numpy()] if len(nu) else np.zeros((0, X.shape[1]), np.float32)
    tr = ((cards["split"] == "train") & cards["source"].isin(["ktas", "nhamcs"])).to_numpy()
    va = ((cards["split"] == "val") & (cards["source"] == "ktas")).to_numpy()
    Xtr = np.vstack([X[tr], Xu])
    ytr = np.concatenate([cards.loc[tr, "urgent"].to_numpy(), nu["urgent"].astype(int).to_numpy()])
    y = cards["urgent"].to_numpy()
    best = max(C_GRID, key=lambda C: roc_auc_score(
        y[va], make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(Xtr, ytr).predict_proba(X[va])[:, 1]))
    cand = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=5000)).fit(Xtr, ytr)
    p_va = cand.predict_proba(X[va])[:, 1]
    cand_thr = {s: float(np.quantile(p_va[y[va] == 1], 1 - s)) for s in (0.8, 0.9)}
    report["urgency"] = {"current": _urgency_metrics(current["head"], cur_thr, cards, X),
                         "candidate": _urgency_metrics(cand, cand_thr[0.9], cards, X),
                         "new_training_patients": int(len(nu)), "C": best}

    # --- pathway flagger: same fit_flagger, original labels + in-use labels (patients with a final diagnosis)
    lab = pd.read_parquet(PROCESSED / "pathway_labels.parquet")
    from config import OUTPUTS
    e = np.load(OUTPUTS / "embeddings" / "cards_pool1__base4bit.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(e["ids"])}
    lab = lab[lab["id"].isin(pos)].reset_index(drop=True)
    XP = e["X"][[pos[i] for i in lab["id"]]]
    nd = new[new["has_dx"]]
    XP_all = np.vstack([XP, Xn[nd.index.to_numpy()]]) if len(nd) else XP
    cc = lab["chief_complaint"].fillna("").astype(str).tolist() + nd["chief_complaint"].tolist()
    Y = np.vstack([lab[ph.NAMES].to_numpy(), nd[ph.NAMES].astype(int).to_numpy()]) if len(nd) else lab[ph.NAMES].to_numpy()
    split = np.concatenate([lab["split"].to_numpy(), nd["split"].to_numpy()])
    cand_fl = ph.fit_flagger(XP_all, cc, Y, (split == "train") | (split == "use_train"), split == "val")
    cur_fl = joblib.load(heads_dir / LIVE["pathways"])
    te = split == "test"
    cc_te = [c for c, t in zip(cc, te) if t]
    report["pathways"] = {"current": _pathway_metrics(cur_fl, XP_all[te], cc_te, Y[te]),
                          "candidate": _pathway_metrics(cand_fl, XP_all[te], cc_te, Y[te]),
                          "new_training_patients": int((nd["split"] == "use_train").sum())}
    chk = split == "use_check"                      # patients from use the candidate never trained on
    if chk.sum() >= 5:
        cc_chk = [c for c, t in zip(cc, chk) if t]
        report["pathways"]["in_use_check"] = {"n": int(chk.sum()),
                                              "current": _pathway_metrics(cur_fl, XP_all[chk], cc_chk, Y[chk]),
                                              "candidate": _pathway_metrics(cand_fl, XP_all[chk], cc_chk, Y[chk])}

    # --- the gate (fixed in advance), on the original test patients
    u, pw, fails = report["urgency"], report["pathways"], []
    for src in ("ktas", "nhamcs"):
        if u["candidate"][src]["auroc"] < u["current"][src]["auroc"] - GATE["urgency_auroc_drop_max"]:
            fails.append(f"urgency AUROC fell on {src}")
        if u["candidate"][src]["sensitivity"] < u["current"][src]["sensitivity"] - GATE["urgency_sensitivity_drop_max"]:
            fails.append(f"fewer urgent {src} patients flagged")
    if pw["candidate"]["caught_mean"] < pw["current"]["caught_mean"] - GATE["pathway_caught_drop_max"]:
        fails.append("pathways caught fell")
    if pw["candidate"]["false_flag"] > pw["current"]["false_flag"] + GATE["pathway_false_flag_rise_max"]:
        fails.append("more false pathway flags")
    report["gate_failures"] = fails
    if fails or dry_run:
        report["result"] = ("dry run: gate " + ("failed: " + "; ".join(fails) if fails else "passed")) if dry_run \
            else "gate failed, kept the current models: " + "; ".join(fails)
        return _finish(report, out_dir)

    # --- switch: keep the old files, write the new ones
    stamp = time.strftime("%Y%m%d_%H%M%S")
    keep = heads_dir / "versions" / stamp
    keep.mkdir(parents=True, exist_ok=True)
    for f in LIVE.values():
        shutil.copy2(heads_dir / f, keep / f)
    joblib.dump({**current, "head": cand, "threshold": cand_thr[0.8], "thresholds": cand_thr,
                 "learned_from_use": len(nu)}, heads_dir / LIVE["urgent"])
    joblib.dump(cand_fl, heads_dir / LIVE["pathways"])
    report.update(switched=True, previous_models=str(keep), result="gate passed, new models in use (restart the app)")
    return _finish(report, out_dir)


def _finish(report, out_dir):
    ups = _read_updates()
    ups.append(report)
    (out_dir / "updates.json").write_text(json.dumps(ups, indent=1))
    with open(out_dir / "CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## {report['time']}: {report['result']}\n")
        for part in ("urgency", "pathways"):
            if part in report:
                f.write(f"- {part}: current {report[part]['current']} -> candidate {report[part]['candidate']}\n")
    return report


def rollback():
    heads_dir, _ = _paths()
    versions = sorted((heads_dir / "versions").glob("*")) if (heads_dir / "versions").exists() else []
    if not versions:
        return "no earlier version kept"
    last = versions[-1]
    for f in LIVE.values():
        shutil.copy2(last / f, heads_dir / f)
    shutil.move(str(last), str(last) + "_rolled_back")
    ups = _read_updates()
    ups.append({"time": time.strftime("%Y-%m-%d %H:%M"), "result": f"rolled back to {last.name}", "switched": False})
    (_paths()[1] / "updates.json").write_text(json.dumps(ups, indent=1))
    return f"restored the models kept in {last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", default=str(HERE.parent / "web" / "case_files" / "learning"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()
    if a.rollback:
        print(rollback())
    elif a.update:
        r = update(a.feedback, a.dry_run)
        print(json.dumps({k: v for k, v in r.items() if k != "gate"}, indent=1))
    else:
        print(json.dumps(FeedbackLog(a.feedback).summary(), indent=1))


if __name__ == "__main__":
    main()
