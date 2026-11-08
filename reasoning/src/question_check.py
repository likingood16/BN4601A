"""Question-writer test (GPU, 8 Nov 2026): the current text adapter vs the trained questions adapter on held-out Pool 1
test patients (make_question_data.py --> questions_test.jsonl; never trained on).

    python question_check.py --n 150      # -> outputs/metrics/question_check.json

Per answer: valid JSON with questions; asks for something the notes already have (worse); junk (identity details or a
1-2 word fragment, e.g. "LSTC", "full name"); for patients who really needed a pathway (final diagnosis), whether a
question is about that pathway. Scored by word topics below, the same for both adapters.
"""
import argparse
import json
import re

import pandas as pd

from answers import final_answer
from charting import ASKS_FOR, parse_lists
from config import OUTPUTS, PROCESSED
from model import ask_messages

KNOWN_TOPIC = dict(ASKS_FOR, onset=r"when did (?:it|this|the \w+) start|onset", allergy=r"allerg", meds=r"medication|medicine")
JUNK = r"\bname\b|address|phone|date of birth|\bnric\b|insurance|next of kin|occupation"
TOPIC = {"chest pain": r"radiat|crushing|sharp|tearing|heart disease|cardiac|risk factor|sweat|chest pain",
         "possible stroke": r"last (?:known|seen)|glucose|blood thinner|anticoag|weakness|speech|face",
         "breathing": r"asthma|copd|heart failure|breath|wheez|leg swelling|cough",
         "collapse/syncope": r"witness|warning|palpitation|collapse|faint|injur|head",
         "seizure": r"seizure|epilep|fit|how long|first",
         "possible poisoning": r"what was taken|how much|substance|on purpose|self-harm|overdose|tablet|ingest",
         "possible infection": r"infection|fever|urine|cough|immunosupp|chemo|steroid|wound|skin",
         "trauma": r"mechanism|fall|speed|blood thinner|injur|hit|head"}


def score(raw, meta):
    qs, todo = parse_lists(raw)
    low = [q.lower() for q in qs]
    known = set(meta["known"])
    asks_known = any(any(re.search(KNOWN_TOPIC[k], q) for k in known if k in KNOWN_TOPIC)
                     and not any(re.search(r, q) for k, r in KNOWN_TOPIC.items() if k not in known) for q in low)
    junk = any(re.search(JUNK, q) or len(re.findall(r"[a-z]{2,}", q)) < 2 for q in low)   # "age?", "LSTC"
    hit = {p: any(re.search(TOPIC[p], q) for q in low) for p in meta["true_paths"] if p in TOPIC}
    return {"valid": bool(qs), "asks_known": asks_known, "junk": junk, "pathway_hit": hit, "n_q": len(qs), "n_todo": len(todo),
            "questions": qs, "todo": todo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    from pipeline import EDPipeline
    ed = EDPipeline()
    rows = pd.read_json(PROCESSED / "questions_test.jsonl", lines=True)
    rows["n_paths"] = rows["meta"].apply(lambda m: len(json.loads(m)["true_paths"]))
    rows = pd.concat([rows[rows.n_paths > 0].sample(a.n // 2, random_state=0),     # half with a real pathway
                      rows[rows.n_paths == 0].sample(a.n - a.n // 2, random_state=0)])
    arms = {"current (task adapter)": ed.text_adapter, "trained questions adapter": ed.step_adapter.get("questions")}
    res = {"n": len(rows), "arms": {}}
    for arm, adapter in arms.items():
        if adapter is None:
            print("no", arm); continue
        out = []
        for _, r in rows.iterrows():
            msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in r["messages"][:2]]
            ed.model.set_adapter(adapter)
            raw = final_answer(ask_messages(ed.model, ed.processor, msgs, 600, False)) or ""
            out.append(score(raw, json.loads(r["meta"])) | {"id": r["id"]})
        hits = [v for o in out for v in o["pathway_hit"].values()]
        res["arms"][arm] = {"valid": round(sum(o["valid"] for o in out) / len(out), 3),
                            "asks_for_known": round(sum(o["asks_known"] for o in out) / len(out), 3),
                            "junk": round(sum(o["junk"] for o in out) / len(out), 3),
                            "pathway_question_asked": round(sum(hits) / max(len(hits), 1), 3),
                            "examples": out[:12]}
        print(arm, {k: v for k, v in res["arms"][arm].items() if k != "examples"}, flush=True)
    (OUTPUTS / "metrics" / "question_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
