"""How should the urgent flag and the level combine? Over-triage check on the held-out TEST patients (7 Nov 2026).

    python triage_policy.py

The app today: level = level model, but at most 2 whenever the patient is flagged (head at the 90% cut-off OR the
vitals safety floor). The real-AI check showed this pushes many patients one level too urgent. Policies compared:
  A  current:      flagged -> level min(model, 2), cut-off 90%
  B  80% cut-off:  same rule, the head's 80% cut-off (fewer flags)
  C  separate:     level from the level model alone; the flag is shown next to it, not forced onto the level
  D  floor only:   level forced to <=2 only by the vitals safety floor or when the level model's set includes 1-2
Per policy: urgent caught (true 1-2 given level 1-2), under-triage, exact, within one, over-triage (more urgent than
true). Saves outputs/metrics/triage_policy.json.
"""
import json

import joblib
import numpy as np
import pandas as pd

from config import HEADS_DIR, OUTPUTS, PROCESSED

SOURCES = ("ktas", "nhamcs")


def main():
    z = np.load(OUTPUTS / "embeddings" / "cards_pool1__base4bit.npz", allow_pickle=True)
    emb = pd.DataFrame({"id": z["ids"], "row": range(len(z["ids"]))})
    pool = pd.read_parquet(PROCESSED / "pool1.parquet")
    heads = pd.read_parquet(PROCESSED / "head_scores.parquet")[["id", "p_urgent", "floor_flag"]]
    test = pool[pool["split"] == "test"].merge(emb, on="id").merge(heads, on="id")
    X = z["X"][test["row"].to_numpy()]
    level = joblib.load(HEADS_DIR / "level__base4bit.joblib")
    urgent = joblib.load(HEADS_DIR / "urgent__base4bit__ktas_nhamcs.joblib")
    cut = urgent.get("thresholds", {0.8: urgent["threshold"]})
    probs = level["model"].predict_proba(X)
    lv = probs.argmax(1) + 1
    in_set_urgent = (probs[:, :2] >= 1 - level["conformal_cutoff"]).any(1)
    p, floor = test["p_urgent"].to_numpy(), test["floor_flag"].to_numpy(bool)
    true = test["triage_level"].to_numpy()

    policies = {
        "A_current_90": np.where((p >= cut[0.9]) | floor, np.minimum(lv, 2), lv),
        "B_cutoff_80": np.where((p >= cut[0.8]) | floor, np.minimum(lv, 2), lv),
        "C_separate_flag": lv,
        "D_floor_or_set": np.where(floor | in_set_urgent, np.minimum(lv, 2), lv),
    }
    res = {"n_test": len(test), "cut_offs": {str(k): v for k, v in cut.items()}}
    for name, assigned in policies.items():
        out = {}
        for src in SOURCES + ("all",):
            m = np.ones(len(test), bool) if src == "all" else (test["source"] == src).to_numpy()
            t, a = true[m], assigned[m]
            urg = t <= 2
            out[src] = {"n": int(m.sum()), "urgent_caught": round(float((a[urg] <= 2).mean()), 3),
                        "under_triage": round(float((a[urg] > 2).mean()), 3),
                        "exact": round(float((a == t).mean()), 3), "within_1": round(float((abs(a - t) <= 1).mean()), 3),
                        "over_triage": round(float((a < t).mean()), 3), "share_level_1_2": round(float((a <= 2).mean()), 3)}
        res[name] = out
    (OUTPUTS / "metrics" / "triage_policy.json").write_text(json.dumps(res, indent=1))
    rows = [(n, s, *res[n][s].values()) for n in policies for s in SOURCES + ("all",)]
    print(pd.DataFrame(rows, columns=["policy", "source", "n", "urgent_caught", "under", "exact", "within1", "over",
                                      "share_L1-2"]).to_string(index=False))


if __name__ == "__main__":
    main()
