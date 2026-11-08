"""Training data for the AI question writer (8 Nov 2026, branch ai-reasoning).

    python make_question_data.py        # -> data/processed/questions_{train,val,test}.jsonl (laptop, CPU, ~1 min)

Why: in the full-app check the 4B model's own questions were often useless ("LSTC", "full name", "age?" with the age
on the card). This teaches it, with a small LoRA adapter used only for the questions step, what an ED clinician asks.

One example per Pool 1 patient (train/val/test = Pool 1's split, so no test patient is trained on):
  input  = what the app gives the question step: the patient line, the AI pathway flags (pathway_heads.PathwayFlagger,
           the real ones for this patient, sometimes plus a wrong one) and the case file built from a messy note
           (charting_check.note_for/messier; some values, allergies, medicines and onset written, some not)
  answer = {"questions": [...], "todo": [...]}:
           1. the questions for each pathway the patient REALLY needed (answer key = final diagnosis, pathway_heads.py),
              so the model learns to ask about the real problem and to ignore a flag the note doesn't support
           2. important values the note doesn't have (BP, heart rate, ...), never ones it has
           3. onset / allergies / medicines when not written
           to-dos = the standard first work-up of each real pathway, and repeat observations if values are missing.
The question bank (charting.PATHWAY_QUESTIONS) is the clinician wording; which questions apply is learned from data.
"""
import json
import random

import joblib

import numpy as np
import pandas as pd

import pathway_heads
from charting import MARK, PATHWAY_QUESTIONS, QUESTION_SYSTEM, TRIAGE, _test_report
from charting_check import messier, note_for
from config import HEADS_DIR, OUTPUTS, PROCESSED, SEED
from data import load_pool1

TODOS = {"chest pain": ["ECG within 10 minutes", "Troponin"],
         "possible stroke": ["Capillary glucose", "Urgent CT brain (stroke team if within window)"],
         "breathing": ["Oxygen saturation and work of breathing", "Chest X-ray"],
         "collapse/syncope": ["12-lead ECG", "Capillary glucose"],
         "seizure": ["Capillary glucose", "Time the seizure, protect the airway"],
         "possible poisoning": ["Paracetamol and salicylate levels", "ECG"],
         "possible infection": ["Sepsis screen (qSOFA)", "Blood cultures and lactate"],
         "trauma": ["X-ray of the injured area", "Check tetanus status"]}
QBANK = {"breathing": "breathing complaint"}                       # name in PATHWAY_QUESTIONS when it differs
MISSING_Q = [("sbp", "What is the blood pressure?"), ("hr", "What is the heart rate?"),
             ("spo2", "What is the oxygen saturation?"), ("mental", "What is the level of consciousness (AVPU/GCS)?"),
             ("temp_c", "What is the temperature?"), ("rr", "What is the respiratory rate?")]
ONSET_KNOWN = {"When did the pain start, and what is it like (crushing, sharp, tearing)?":
               "What is the pain like (crushing, sharp, tearing)?"}
ONSET = ["started 2 hours ago", "since this morning", "onset yesterday", "for 3 days", "sudden onset 30 min ago"]
ALLERGY = ["NKDA", "allergic to penicillin", "no known allergies"]
MEDS = ["on metformin", "meds: amlodipine", "on warfarin", "no regular meds", "takes aspirin daily"]


def example(row, true_paths, flags, rng):
    keep = row.copy()
    for c in ("hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain", "mental"):   # leave some values out
        if rng.random() < 0.15:
            keep[c] = np.nan
    note = note_for(keep, rng)
    if rng.random() < 0.5:
        note = messier(note, rng)
    has = {k: rng.random() < 0.35 for k in ("onset", "allergy", "meds")}
    extra = [rng.choice(ONSET) if has["onset"] else "", rng.choice(ALLERGY) if has["allergy"] else "",
             rng.choice(MEDS) if has["meds"] else ""]
    if row.get("injured") == 1:                                  # the note must show it, or trauma questions come from nowhere
        extra.insert(0, rng.choice(["injured", "post injury", "trauma", "injury today"]))
    note = ". ".join([note] + [e for e in extra if e])
    report = _test_report("", [note], []).replace(f"_{MARK} sorted by keyword rules, MedGemma not loaded._\n\n", "")
    report = report.split(f"## {TRIAGE}")[0].strip()
    sex = keep["sex"] if isinstance(keep["sex"], str) else "sex not recorded"
    arrival = {1: "ambulance", 0: "walk-in"}.get(keep["ambulance"], "not recorded")
    age = int(keep["age"]) if keep["age"] == keep["age"] else "not recorded"
    context = f"PATIENT: age {age}, {sex}, arrival {arrival}.\nPathway flags: " + ("; ".join(flags) if flags else "none")
    questions, todo = [], []
    for p in true_paths:
        bank = PATHWAY_QUESTIONS.get(QBANK.get(p, p), [])[:2]
        if has["onset"]:                                         # the note gives the onset: don't ask it again
            bank = [ONSET_KNOWN.get(q, q) for q in bank]
        questions += bank
        todo += TODOS.get(p, [])
    missing = [q for c, q in MISSING_Q if keep[c] != keep[c]]
    questions += missing[:2]
    if not has["onset"]:
        questions.append("When did it start?")
    if not has["allergy"]:
        questions.append("Any allergies?")
    if not has["meds"]:
        questions.append("Regular medications (especially blood thinners)?")
    if missing:
        todo.append("Repeat full set of observations")
    dedup = lambda xs: list(dict.fromkeys(xs))
    answer = json.dumps({"questions": dedup(questions)[:4], "todo": dedup(todo)[:4]})
    meta = {"known": [c for c in ("age", "sex", "hr", "sbp", "rr", "spo2", "temp_c", "nrs_pain", "mental")
                      if keep[c] == keep[c] and keep[c] is not None] + [k for k, v in has.items() if v],
            "true_paths": true_paths, "flags": flags}
    return {"source": "task_questions", "id": row["id"], "meta": json.dumps(meta),
            "messages": [{"role": "system", "content": QUESTION_SYSTEM},
                         {"role": "user", "content": f"{context}\n\n{report}"},
                         {"role": "assistant", "content": answer}]}


def main():
    rng = random.Random(SEED)
    pool = load_pool1()
    lab = pd.read_parquet(PROCESSED / "pathway_labels.parquet").set_index("id")
    e = np.load(OUTPUTS / "embeddings" / "cards_pool1__base4bit.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(e["ids"])}
    pool = pool[pool["id"].isin(pos) & pool["id"].isin(lab.index)].reset_index(drop=True)
    flagger = joblib.load(HEADS_DIR / "pathways__base4bit.joblib")
    F = flagger.flag_matrix(e["X"][[pos[i] for i in pool["id"]]], pool["chief_complaint"].fillna("").astype(str).tolist())
    texts = [h["flag"] for h in flagger.heads.values()]
    counts = {}
    for split in ("train", "val", "test"):
        rows = []
        for k in np.flatnonzero((pool["split"] == split).to_numpy()):
            row = pool.iloc[k]
            true_paths = [n for n in pathway_heads.NAMES if lab.at[row["id"], n] == 1]
            if row.get("injured") == 1:
                true_paths.append("trauma")
            flags = [t for t, on in zip(texts, F[k]) if on]
            if rng.random() < 0.25:                              # a wrong flag sometimes, as in the app (31%)
                flags = list(dict.fromkeys(flags + [rng.choice(texts)]))
            rows.append(example(row, true_paths, flags, rng))
        if split == "val":
            rows = rows[:300]                                    # enough to watch the val loss
        with open(PROCESSED / f"questions_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts[split] = len(rows)
    print(counts)
    print(json.dumps(rows[0]["messages"][1:], indent=1)[:2500])


if __name__ == "__main__":
    main()
