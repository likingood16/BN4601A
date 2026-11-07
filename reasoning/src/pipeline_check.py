"""Tests of the joined-up pipeline. Each part saves outputs/metrics/pipeline_check_<part>.json.

    python pipeline_check.py --part record            # no model needed: the record is built correctly
    python pipeline_check.py --part scores  --n 50    # same urgency + level as the saved results?
    python pipeline_check.py --part results --n 20    # do the chat and SBAR use results added during the visit?
    python pipeline_check.py --part images            # Base vs Adapted vs the detector on the same 463 test X-rays
    --n 2 is the quick laptop trial. The images part continues from its last saved X-ray if restarted.

"results" uses made-up lab values: it shows that results get USED, not that MedGemma interprets them correctly.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

from answers import final_answer, has_sbar_headings
from cards import assess, dev_vitals_model, make_card
from config import OUTPUTS, PROCESSED, REPO, SEED
from data import load_pool1
from ed_assessment import ed_assessment
from prompts import TURNS
from visit import END, Decision, Visit

DETECTOR_DATA = Path(REPO / "vision" / "detector" / "data" / "yolo")
# (section, made-up result, words that show the answer used it)
LATER_RESULTS = [
    ("bloods", "Troponin T 2.1 ng/mL (high).", r"troponin"),
    ("ecg", "ST elevation in leads V1-V4.", r"st[- ]elevation|stemi"),
    ("bloods", "Lactate 5.2 mmol/L (high).", r"lactate"),
    ("bedside", "Finger-prick glucose 2.1 mmol/L (low).", r"glucose|hypoglyc|blood sugar"),
    ("imaging", "CT head report: acute subdural haematoma.", r"subdural"),
    ("bloods", "Potassium 6.8 mmol/L (high).", r"potassium|hyperkal"),
]


def save(part, res):
    path = OUTPUTS / "metrics" / f"pipeline_check_{part}.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1), f"\nsaved {path}")


def test_patients(n):
    pool = load_pool1()
    return pool[pool["split"] == "test"].sample(n, random_state=SEED)


def check_record():
    """Later results reach the record; the arrival card and the decision stay exactly as they were."""
    row = test_patients(1).iloc[0].to_dict()
    vm = dev_vitals_model()
    out = assess(vm, row)
    card = make_card(row, out)
    d = Decision(0.5, True, False, True, 2, (1, 2), out["floor_priority"], out["news2"])
    v = Visit(card=card, decision=d, ed=ed_assessment(row))
    empty = v.record()
    for key, text in [("allergies", "penicillin"), ("bloods", "Troponin 2.1"), ("bloods", r"Lactate 5 \ high"),
                      ("imaging", "bone X-ray: nothing marked"), ("treatment", "aspirin 300 mg")]:
        v.add(key, text)
    full = v.record()
    checks = {
        "card_unchanged": v.card == card and "Troponin" not in v.card,
        "decision_unchanged": v.decision == d,
        "empty_section_says_not_recorded": "BLOOD TESTS: not recorded" in empty,
        "both_blood_results_kept": "BLOOD TESTS: Troponin 2.1 | Lactate 5 \\ high" in full,
        "existing_line_filled_not_repeated": full.count("ALLERGIES:") == 1 and "ALLERGIES: penicillin" in full,
        "imaging_replaces_none": "IMAGING: bone X-ray" in full and "IMAGING: none" not in full,
        "ed_assessment_included": "ED ASSESSMENT" in full and "Pathway flags" in full,
        "decision_on_record": "URGENCY DECISION" in full and "TRIAGE LEVEL (trained model): 2" in full,
        "ends_properly": full.endswith(END) and full.count(END) == 1,
    }
    print(full)
    try:
        v.add("nonsense", "x")
        checks["unknown_section_refused"] = False
    except ValueError:
        checks["unknown_section_refused"] = True
    # the ED section must work for every patient, whatever is missing
    pool = load_pool1()
    for r in pool.sample(min(2000, len(pool)), random_state=SEED).to_dict("records"):
        ed_assessment(r)
    checks["ed_assessment_runs_on_2000_patients"] = True
    if not all(checks.values()):
        raise SystemExit(f"FAILED: {checks}")
    save("record", {"passed": all(checks.values()), "checks": checks})


def check_scores(n):
    """The new flow must give the same answers as the results already reported, on the same patients."""
    from heads import load_xy
    from pipeline import EDPipeline

    ed = EDPipeline(vitals_model=dev_vitals_model(), sensitivity=0.8)     # the saved results used the 80% cut-off
    cards = pd.read_parquet(PROCESSED / "cards_pool1.parquet").set_index("id")
    saved = pd.read_parquet(PROCESSED / "head_scores.parquet").set_index("id")
    table, X = load_xy("base4bit")
    pos = {i: k for k, i in enumerate(table["id"])}
    rows = []
    for row in test_patients(n).to_dict("records"):
        i = row["id"]
        v = ed.triage(row, explain=False)
        probs = ed.level_model.predict_proba(X[[pos[i]]])[0]
        rows.append({"id": i, "card_same": v.card == cards.loc[i, "card"], "p_new": v.decision.p_urgent,
                     "p_saved": float(saved.loc[i, "p_urgent"]), "flag_new": v.decision.flagged,
                     "flag_saved": bool(saved.loc[i, "flagged"]), "level_new": v.decision.level,
                     "level_saved": int(min(probs.argmax() + 1, 2) if saved.loc[i, "flagged"] else probs.argmax() + 1)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    res = {"n": len(t), "cards_identical": int(t["card_same"].sum()),
           "largest_score_difference": float((t["p_new"] - t["p_saved"]).abs().max()),
           "same_flag": int((t["flag_new"] == t["flag_saved"]).sum()),
           "same_level": int((t["level_new"] == t["level_saved"]).sum()),
           "cases": rows}
    save("scores", res)


def check_results(n, with_image=True):
    """Consultation with one result arriving mid-way (same 4 turns as answers.py)."""
    from pipeline import NO_ANSWER, EDPipeline

    ed = EDPipeline(vitals_model=dev_vitals_model())
    cases = []
    for k, row in enumerate(test_patients(n).to_dict("records")):
        section, result, words = LATER_RESULTS[k % len(LATER_RESULTS)]
        times = {}
        v = ed.triage(row)
        times.update(ed.times)
        before = (v.card, v.decision)
        image_text = None
        if with_image and k < 3:                    # the whole flow, including one image
            image = sorted((DETECTOR_DATA / "images" / "test").glob("*.jpg"))[k]
            image_text = ed.read_image(v, image, "bone_xray")["text"]
            times.update(image=ed.times["image"], detector=ed.times["detector"])
        first = ed.chat(v, TURNS[0])
        times["chat_1"] = ed.times["chat"]
        v.add(section, result)
        second = ed.chat(v, "A new result has come back. Does this change your assessment?")
        times["chat_2"] = ed.times["chat"]
        lay = ed.chat(v, TURNS[2])
        times["chat_3"] = ed.times["chat"]
        sbar = ed.sbar(v)
        times["sbar"] = ed.times["sbar"]
        low = sbar.lower()
        cases.append({
            "id": row["id"], "flagged": v.decision.flagged, "level": v.decision.level, "result": result, "image_text": image_text,
            "explanation_valid": bool(v.explanation.get("valid_json")), "explanation_cut_off": v.explanation.get("cut_off"),
            "decision_unchanged": before == (v.card, v.decision),
            "chat_used_result": bool(re.search(words, second, re.I)), "sbar_used_result": bool(re.search(words, sbar, re.I)),
            "sbar_headings": has_sbar_headings(sbar),
            "flagged_called_non_urgent": v.decision.flagged and bool(re.search(r"non-?urgent|level [345]\b", low)),
            "no_answer": NO_ANSWER in (first, second, lay, sbar),
            "sbar_words": len(sbar.split()), "times": times,
            "first": first, "second": second, "lay": lay, "sbar": sbar, "explanation": v.explanation})
        print({c: cases[-1][c] for c in ("id", "flagged", "chat_used_result", "sbar_used_result", "sbar_headings", "times")},
              flush=True)
    t = pd.DataFrame(cases)
    res = {"n": len(t), **{c: int(t[c].sum()) for c in ("explanation_valid", "decision_unchanged", "chat_used_result",
                                                        "sbar_used_result", "sbar_headings", "flagged_called_non_urgent",
                                                        "no_answer")},
           "n_flagged": int(t["flagged"].sum()), "median_sbar_words": float(t["sbar_words"].median()),
           # seconds per step: median and slowest over all cases
           "seconds": {k: {"median": float(x.median()), "max": float(x.max())}
                       for k, x in pd.DataFrame(list(t["times"])).items()},
           "cases": cases}
    save("results", res)


def check_images(limit=None):
    """Same 463 X-rays, same question, Base vs Adapted (the brief's comparison) next to the detector, one at a time.
    The adapted model was trained to write captions, so it may not answer YES/NO: we also count a fracture
    named in its plain description."""
    from image_reader import describe_image, detect_fracture, image_messages, load_image_model
    from model import ask_messages
    from prompts import FRACTURE_QUESTION

    split = pd.read_csv(DETECTOR_DATA / "splits.csv")
    split = split[split["split"] == "test"]
    if limit:                       # a few of each kind
        split = split.groupby("fractured").head(limit)
    res = {"n_fractured": int(split["fractured"].sum()), "n_normal": int((~split["fractured"]).sum())}
    for name, adapted in (("base", False), ("adapted", True)):
        model, processor = load_image_model(adapted=adapted)
        log = OUTPUTS / "metrics" / f"pipeline_check_images_{name}.jsonl"
        done = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() and not limit else []
        seen = {d["image_id"] for d in done}
        with log.open("a") as f:
            for r in split.to_dict("records"):
                if r["image_id"] in seen:
                    continue
                path = DETECTOR_DATA / "images" / "test" / r["image_id"]
                answer = final_answer(ask_messages(model, processor, image_messages(path, FRACTURE_QUESTION), 512))
                said = re.findall(r"ANSWER:\s*\**\s*(YES|NO)", answer, re.I)
                text = describe_image(model, processor, path, "bone_xray")
                row = {"image_id": r["image_id"], "fractured": bool(r["fractured"]), "said": said[-1].upper() if said else None,
                       "answer": answer, "description": text}
                done.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(name, len(done), r["image_id"], r["fractured"], row["said"], text[:80], flush=True)
        t = pd.DataFrame(done)
        t["named"] = t["description"].str.contains("fractur", case=False) & ~t["description"].str.contains(r"\bno\b[^.]*fractur", case=False)
        frac, normal = t[t["fractured"]], t[~t["fractured"]]
        res[name] = {"no_clear_answer": int(t["said"].isna().sum()),
                     "yes_no_sensitivity": float((frac["said"] == "YES").mean()),
                     "yes_no_specificity": float(1 - (normal["said"] == "YES").mean()),
                     "description_sensitivity": float(frac["named"].mean()),
                     "description_specificity": float(1 - normal["named"].mean()),
                     "median_words": float(t["description"].str.split().str.len().median()),
                     "examples": dict(zip(t["image_id"].head(3), t["description"].head(3)))}
        del model, processor
        torch.cuda.empty_cache()
    flags = split["image_id"].map(lambda n: detect_fracture(DETECTOR_DATA / "images" / "test" / n)["n"] > 0)
    res["detector"] = {"sensitivity": float(flags[split["fractured"]].mean()),
                       "specificity": float(1 - flags[~split["fractured"]].mean())}
    save("images", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["record", "scores", "results", "images"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--no-image", action="store_true", help="results part only: a 4 GB laptop can't hold both models")
    a = ap.parse_args()
    if a.part == "record":
        check_record()
    elif a.part == "scores":
        check_scores(a.n or 50)
    elif a.part == "results":
        check_results(a.n or 20, with_image=not a.no_image)
    else:
        check_images(a.n)
