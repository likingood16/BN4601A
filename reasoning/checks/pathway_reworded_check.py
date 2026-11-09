"""Reworded-notes test for the AI pathway flags (GPU; 8 Nov 2026).

    python pathway_reworded_check.py

Test patients who needed each pathway (answer key from their final diagnosis, pathway_heads.py) get their complaint
replaced by a phrase that means the same but avoids the old word rules' words ("passed out at church", "fitting at
home"). Then the full app step runs (EDPipeline.triage: card -> MedGemma vector -> pathway heads) and we count how
often the right pathway is flagged, by the AI heads and by the old word rules (ed_assessment.PATHWAYS). Controls: the
same patients with their real complaint, and test patients who needed none of the pathways (false flags).
Saves outputs/metrics/pathways_reworded.json.
"""
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702  (reasoning/src etc. importable)
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("src", "training", "checks")]
import json
import random
import re

import pandas as pd

from config import OUTPUTS, PROCESSED
from data import load_pool1
from ed_assessment import PATHWAYS as OLD_RULES
from pathway_heads import NAMES, PATHWAYS

REWORDED = {
    "chest pain": ["heaviness behind the breastbone going to the left arm", "crushing feeling in the middle of chest",
                   "chest feels heavy and sweaty"],
    "possible stroke": ["can't move left arm, face drooping", "words coming out wrong, right side floppy",
                        "sudden left side gone dead"],
    "breathing": ["gasping, can't finish sentences", "lips blue and panting", "air hunger since this morning"],
    "collapse/syncope": ["passed out at church", "blacked out while standing", "found on the floor, came round after a minute"],
    "seizure": ["fitting at home, shaking all over", "jerking all limbs then very sleepy", "had a fit in the shop"],
    "possible poisoning": ["took a bottle of her pills", "swallowed bleach", "drank weedkiller after an argument"],
    "possible infection": ["burning up and shaking", "hot to touch with rigors", "high temperature for 3 days, shivering"],
}
PER_PATHWAY = 15
SEED = 42


def flags_of(ed_text):
    m = re.search(r"Pathway flags: (.*)", ed_text or "")
    got = m.group(1).lower() if m else ""
    return {n: PATHWAYS[n][0].split(" ->")[0].lower() in got for n in NAMES}


def rules_of(complaint):
    return {n: bool(re.search(w, complaint.lower())) for n, (w, _) in zip(NAMES, OLD_RULES)}


def main():
    from pipeline import EDPipeline
    ai = EDPipeline(explainer=None)
    rng = random.Random(SEED)
    lab = pd.read_parquet(PROCESSED / "pathway_labels.parquet").set_index("id")
    pool = load_pool1()
    test = pool[pool["split"] == "test"].set_index("id")
    res = {"per_pathway": {}, "per_pathway_n": PER_PATHWAY}

    def run(row, complaint):
        p = {k: (None if v != v else v) for k, v in row.items()}
        p["chief_complaint"] = complaint
        return flags_of(ai.triage(p, explain=False).ed)

    for name in NAMES:
        ids = [i for i in test.index if i in lab.index and lab.loc[i, name] == 1]
        ids = rng.sample(ids, min(PER_PATHWAY, len(ids)))
        ai_orig = ai_new = rule_orig = rule_new = 0
        for i in ids:
            row = test.loc[i].to_dict()
            phrase = rng.choice(REWORDED[name])
            orig = str(row.get("chief_complaint") or "")
            ai_orig += run(row, orig)[name]
            ai_new += run(row, phrase)[name]
            rule_orig += rules_of(orig)[name]
            rule_new += rules_of(phrase)[name]
        n = max(1, len(ids))
        res["per_pathway"][name] = {"n": len(ids), "ai_real_complaint": round(ai_orig / n, 3),
                                    "ai_reworded": round(ai_new / n, 3), "rules_real_complaint": round(rule_orig / n, 3),
                                    "rules_reworded": round(rule_new / n, 3)}
        print(name, res["per_pathway"][name], flush=True)
    none = [i for i in test.index if i in lab.index and lab.loc[i, NAMES].sum() == 0]
    none = rng.sample(none, min(60, len(none)))
    false_ai = [sum(run(test.loc[i].to_dict(), str(test.loc[i, "chief_complaint"] or "")).values()) for i in none]
    res["no_pathway_patients"] = {"n": len(none), "ai_mean_flags": round(sum(false_ai) / len(none), 3),
                                  "ai_share_with_any_flag": round(sum(f > 0 for f in false_ai) / len(none), 3)}
    print(res["no_pathway_patients"], flush=True)
    (OUTPUTS / "metrics" / "pathways_reworded.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
