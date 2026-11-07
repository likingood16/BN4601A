"""Number extraction test (rubric row "Number extraction"): KTAS patients written up as messy nurse notes, then
`charting.extract_numbers` must get the numbers back. We know the true values, so every field can be scored.

    python charting_check.py --split train     # dev set: look at the misses and fix the rules here
    python charting_check.py --split test      # once, at the end: the number for the report

Each note is built from the patient's real values with a randomly chosen writing style per field (abbreviations,
order, units, Fahrenheit, GCS instead of AVPU, ...). A missing value is simply not written, so we also check that the
rules never invent a number. Scores per field: correct, wrong, missed (written but not found), invented (not written
but found). Saves outputs/metrics/charting_numbers_<split>.json.
"""
import argparse
import json
import random
import re

import numpy as np

from charting import extract_numbers
from config import OUTPUTS
from data import load_pool1

FIELDS = ["age", "sex", "hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain", "mental", "ambulance", "injured"]
GCS = {"alert": [15], "verbal": [13, 12, 10], "pain": [7, 6], "unresponsive": [3]}


def style(rng, options):
    return rng.choice(options)


def note_for(row, rng):
    """One messy note from one patient's real values. Missing values are left out."""
    g = lambda c: None if row[c] != row[c] else row[c]            # NaN -> None
    parts = []
    age, sex = g("age"), g("sex")
    if age is not None and sex:
        a, s = int(age), sex[0].upper()
        parts.append(style(rng, [f"{a}{s}", f"{a} yo {sex}", f"{s} {a}", f"{a}-year-old {'man' if sex == 'male' else 'woman'}",
                                 f"{a}y/o {s}"]))
    parts.append(str(row["chief_complaint"]))
    if g("ambulance") is not None:
        parts.append(style(rng, ["BIBA", "came by ambulance", "brought in by ambulance", "via SCDF ambulance"])
                     if row["ambulance"] == 1 else style(rng, ["walk-in", "walked in", "self-presented", "brought by family"]))
    vit = []
    if g("hr") is not None:
        vit.append(style(rng, ["HR {:g}", "hr {:g}", "PR {:g}", "pulse {:g}", "HR:{:g}"]).format(row["hr"]))
    if g("sbp") is not None and g("dbp") is not None:
        vit.append(style(rng, ["BP {:g}/{:g}", "bp {:g}/{:g}", "BP:{:g}/{:g}", "blood pressure {:g}/{:g}"]).format(row["sbp"], row["dbp"]))
    if g("rr") is not None:
        vit.append(style(rng, ["RR {:g}", "rr {:g}", "resp rate {:g}", "RR:{:g}"]).format(row["rr"]))
    if g("spo2") is not None:
        vit.append(style(rng, ["sats {:g}%", "SpO2 {:g}", "spo2 {:g}% RA", "O2 sats {:g} on air", "sat {:g}%"]).format(row["spo2"]))
    if g("temp_c") is not None:
        t = row["temp_c"]
        vit.append(style(rng, [f"T {t:.1f}", f"temp {t:.1f}C", f"T {t:.1f}°C", f"temp {t * 9 / 5 + 32:.1f}F", f"Temp:{t:.1f}"]))
    if g("nrs_pain") is not None:
        vit.append(style(rng, ["pain {:g}/10", "pain score {:g}/10", "{:g}/10 pain", "NRS {:g}/10"]).format(row["nrs_pain"]))
    if g("mental") is not None:
        m = row["mental"]
        vit.append(style(rng, [{"alert": "alert", "verbal": "responds to voice", "pain": "responds to pain",
                                "unresponsive": "unresponsive"}[m],
                               f"GCS {rng.choice(GCS[m])}", f"AVPU {m[0].upper()}"]))
    rng.shuffle(vit)
    sep = style(rng, [", ", " ", ". ", "; "])
    return sep.join(parts) + ". " + sep.join(vit)


def same(field, truth, got):
    if field == "temp_c":
        return abs(truth - got) <= 0.11                       # Fahrenheit round trip
    if field in ("sex", "mental"):
        return truth == got
    return abs(float(truth) - float(got)) < 1e-6


def messier(note, rng):
    """Real nurse shorthand on top of the template note (--messy): no space before numbers ("hr110", "sat94"),
    'p' for pulse, units stuck on ("120/80mmHg", "110bpm"), a decimal comma ("37,5"), O2 instead of sats, items run
    together without separators, random capitals. Same numbers, so the score is comparable."""
    swaps = [(r"\bPR (\d)", r"p\1"), (r"\bpulse (\d)", r"p \1"), (r"\bsats (\d+)%", r"sat\1"), (r"\bSpO2 (\d)", r"O2 \1"),
             (r"\b(HR|hr) (\d+)", r"\1\2"), (r"\b(BP|bp) (\d+)/(\d+)", r"\1\2/\3mmHg"), (r"\bHR:(\d+)", r"HR \1bpm"),
             (r"(\d+)\.(\d) ?C\b", r"\1,\2"), (r"\b(RR|rr) (\d+)", r"\1=\2"), (r"\bpain (\d+)/10", r"pain \1")]
    for pattern, new in swaps:
        if rng.random() < 0.5:
            note = re.sub(pattern, new, note)
    if rng.random() < 0.3:
        note = re.sub(r"[,;.] (?=[A-Za-z]+ ?\d)", " ", note)          # items run together
    if rng.random() < 0.3:
        note = note.upper()
    return note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--messy", action="store_true", help="add real nurse shorthand (messier)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    df = load_pool1(source="ktas")
    df = df[df["split"] == a.split]
    counts = {f: {"correct": 0, "wrong": 0, "missed": 0, "invented": 0, "written": 0} for f in FIELDS}
    examples = []
    for _, row in df.iterrows():
        note = note_for(row, rng)
        if a.messy:
            note = messier(note, rng)
        got, _ = extract_numbers(note)
        bad = []
        for f in FIELDS:
            truth = row[f]
            written = truth == truth and truth is not None
            if f == "injured":                     # only claimed when an injury word is in the note; never 'no injury'
                if truth == 1:
                    counts[f]["written"] += 1
                    counts[f]["correct"] += got.get(f) == 1
                elif got.get(f) == 1:
                    counts[f]["invented"] += 1; bad.append("injured: claimed but not injured")
                continue
            if f in ("sex", "age") and (row["age"] != row["age"] or not row["sex"]):
                written = False
            if f in ("sbp", "dbp") and (row["sbp"] != row["sbp"] or row["dbp"] != row["dbp"]):
                written = False            # BP is only written when both numbers are known
            c = counts[f]
            if written:
                c["written"] += 1
                if f not in got:
                    c["missed"] += 1; bad.append(f"{f}: missed")
                elif same(f, truth, got[f]):
                    c["correct"] += 1
                else:
                    c["wrong"] += 1; bad.append(f"{f}: {truth} -> {got[f]}")
            elif f in got:
                c["invented"] += 1; bad.append(f"{f}: invented {got[f]}")
        if bad and len(examples) < 40:
            examples.append({"note": note, "problems": bad})
    numeric = ["age", "hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain"]
    summary = {f: round(c["correct"] / c["written"], 3) if c["written"] else None for f, c in counts.items()}
    all_numbers = sum(counts[f]["correct"] for f in numeric) / max(1, sum(counts[f]["written"] for f in numeric))
    wrong_numbers = sum(counts[f]["wrong"] + counts[f]["invented"] for f in numeric)
    res = {"split": a.split, "n_notes": len(df), "vital_numbers_correct": round(all_numbers, 4),
           "vital_numbers_wrong_or_invented": wrong_numbers, "correct_by_field": summary, "counts": counts,
           "examples": examples}
    path = OUTPUTS / "metrics" / f"charting_numbers_{a.split}{'_messy' if a.messy else ''}.json"
    path.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("examples", "counts")}, indent=1))
    for e in examples[:12]:
        print("-", e["problems"], "|", e["note"][:150])
    print("saved", path)


if __name__ == "__main__":
    main()
