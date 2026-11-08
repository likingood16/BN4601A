"""Training data for the AI number reader (8 Nov 2026, fix plan 5o item 3).

    python make_number_data.py        # -> data/processed/numbers_{train,val}.jsonl (laptop, CPU, seconds)

Why: MedGemma read 89% of the numbers in the test notes, and mostly by leaving things out (consciousness, the
diastolic BP). A small LoRA adapter used only for the numbers step is taught the exact answer format.

One note per Pool 1 patient in train (val: 300), built from the patient's real values, with the answer
{"hr": {"value": 112, "quote": "HR 112"}, ...} that charting.extract_numbers_ai expects. Each piece of the note is
written in one of many styles, and the answer quotes exactly the words used, so every label is right by construction.
More styles than charting_check.note_for (narrative sentences, "130 over 85", units stuck on, capitals, numbers
that are NOT vital signs such as doses and bed numbers), so the 190 KTAS test notes stay a fair test.
Test patients are never used.
"""
import json
import random
import re

from charting import INJURY, NUMBERS_SYSTEM
from config import PROCESSED, SEED
from data import load_pool1

GCS = {"alert": [15], "verbal": [14, 13, 12, 10], "pain": [8, 7, 6], "unresponsive": [3]}
DISTRACT = ["bed 12", "on metformin 500mg bd", "PMHx DM2 x10 yrs", "wt 70kg", "last meal 2h ago", "NKDA",
            "lives with wife, 2 children", "LMP 3 weeks ago", "took 2 tabs paracetamol", "hx HTN on amlodipine 5mg",
            "smoker 20 pack yrs", "symptoms x3 days", "rm 4", "tetanus 2019", "aspirin 300mg given by EMS"]


def _g(row, c):
    v = row.get(c)
    return None if v is None or v != v else v


def _num(x):
    return int(x) if float(x).is_integer() else round(float(x), 1)


def pieces(row, rng):
    """[(text, {key: answer item})], one per thing written in the note."""
    out = []
    age, sex = _g(row, "age"), _g(row, "sex")
    if age is not None and sex:
        a, s = int(age), sex[0].upper()
        text = rng.choice([f"{a}{s}", f"{a} yo {sex}", f"{s} {a}", f"{a}-year-old {'man' if sex == 'male' else 'woman'}",
                           f"{a}y/o {s}", f"{a} {sex}", f"{sex}, {a} years", f"{a}yr {s}"])
        out.append((text, {"age": {"value": a, "quote": text}, "sex": {"value": sex, "quote": text}}))
    cc = str(row["chief_complaint"])
    out.append((cc, {}))
    amb = _g(row, "ambulance")
    if amb is not None:
        text = (rng.choice(["BIBA", "came by ambulance", "brought in by ambulance", "via SCDF ambulance", "BIBS",
                            "EMS conveyed", "995 call, ambulance"]) if amb == 1 else
                rng.choice(["walk-in", "walked in", "self-presented", "brought by family", "came in with daughter",
                            "self referred, walked in"]))
        word = re.search(r"BIBA|BIBS|ambulance|EMS|walk-in|walked in|self-presented|brought by family|came in with daughter", text).group(0)
        out.append((text, {"ambulance": {"value": bool(amb == 1), "quote": word}}))
    if _g(row, "injured") == 1:
        m = re.search(INJURY, cc.lower())
        if m:                                   # the complaint already says it
            out[-1 if amb is None else -2][1]["injured"] = {"value": True, "quote": cc[m.start():m.end()]}
        else:
            text = rng.choice(["fell at home", "fall from standing", "MVA", "hit by car", "cut by knife", "twisted ankle, sprain",
                               "assaulted", "dog bite"])
            word = re.search(INJURY, text.lower())
            out.append((text, {"injured": {"value": True, "quote": text[word.start():word.end()]}}))
    hr, sbp, dbp, rr = _g(row, "hr"), _g(row, "sbp"), _g(row, "dbp"), _g(row, "rr")
    spo2, temp, pain, mental = _g(row, "spo2"), _g(row, "temp_c"), _g(row, "nrs_pain"), _g(row, "mental")
    vit = []
    if hr is not None:
        t = rng.choice(["HR {}", "hr {}", "PR {}", "pulse {}", "HR:{}", "p{}", "HR {}bpm", "pulse is {}", "heart rate {}",
                        "P {}", "hr{}"]).format(_num(hr))
        vit.append((t, {"hr": {"value": _num(hr), "quote": t}}))
    if sbp is not None and dbp is not None:
        t = rng.choice(["BP {}/{}", "bp {}/{}", "BP:{}/{}", "blood pressure {}/{}", "BP {}/{}mmHg", "bp{}/{}",
                        "BP {} over {}", "NIBP {}/{}"]).format(_num(sbp), _num(dbp))
        vit.append((t, {"sbp": {"value": _num(sbp), "quote": t}, "dbp": {"value": _num(dbp), "quote": t}}))
    if rr is not None:
        t = rng.choice(["RR {}", "rr {}", "resp rate {}", "RR:{}", "RR={}", "breathing {}/min", "resps {}"]).format(_num(rr))
        vit.append((t, {"rr": {"value": _num(rr), "quote": t}}))
    if spo2 is not None:
        t = rng.choice(["sats {}%", "SpO2 {}", "spo2 {}% RA", "O2 sats {} on air", "sat {}%", "sat{}", "O2 {}",
                        "SpO2 {}% on room air", "saturating {}%"]).format(_num(spo2))
        vit.append((t, {"spo2": {"value": _num(spo2), "quote": t}}))
    if temp is not None:
        f = rng.random() < 0.2
        v = round(temp * 9 / 5 + 32, 1) if f else round(float(temp), 1)
        t = (rng.choice(["temp {}F", "T {} F", "temp {} deg F"]) if f else
             rng.choice(["T {}", "temp {}C", "T {}°C", "Temp:{}", "temp {}", "T {} deg"])).format(v)
        if not f and rng.random() < 0.15:
            t = t.replace(".", ",")              # decimal comma
        vit.append((t, {"temp": {"value": v, "quote": t}, "temp_unit": "F" if f else "C"}))
    if pain is not None:
        t = rng.choice(["pain {}/10", "pain score {}/10", "{}/10 pain", "NRS {}/10", "pain {}", "NRS {}"]).format(_num(pain))
        vit.append((t, {"nrs_pain": {"value": _num(pain), "quote": t}}))
    if mental is not None:
        word = {"alert": "alert", "verbal": "voice", "pain": "pain", "unresponsive": "unresponsive"}[mental]
        options = [({"alert": "alert", "verbal": "responds to voice", "pain": "responds to pain", "unresponsive": "unresponsive"}[mental], word),
                   (f"GCS {rng.choice(GCS[mental])}", None), (f"AVPU {mental[0].upper()}", word)]
        if mental == "verbal":
            options.append(("confused", "confused"))
        if mental == "alert":
            options += [("A&O x3", "alert"), ("alert and orientated", "alert")]
        t, w = rng.choice(options)
        if w is None:                            # GCS: 15 = alert, 13-14 = confused, lower = by the number
            g = int(t.split()[1])
            w = "alert" if g == 15 else "confused" if g >= 13 else "voice" if g >= 9 else "pain" if g >= 4 else "unresponsive"
        vit.append((t, {"consciousness": {"value": w, "quote": t}}))
    rng.shuffle(vit)
    return out, vit


def note_and_answer(row, rng):
    head, vit = pieces(row, rng)
    extra = [(d, {}) for d in rng.sample(DISTRACT, rng.choice([0, 0, 1, 2]))]
    if rng.random() < 0.3:                       # narrative style
        body = ". ".join(t for t, _ in head) + ". On arrival " + ", ".join(t for t, _ in vit) + "."
    else:
        sep = rng.choice([", ", " ", ". ", "; ", " | "])
        body = sep.join(t for t, _ in head + vit)
    if extra:
        body += " " + ". ".join(t for t, _ in extra) + "."
    answer = {}
    for _, items in head + vit:
        answer.update(items)
    if rng.random() < 0.2:                       # all capitals: the quotes follow
        body = body.upper()
        for k, v in answer.items():
            if isinstance(v, dict):
                v["quote"] = v["quote"].upper()
    for k, v in answer.items():
        assert not isinstance(v, dict) or v["quote"] in body, (k, v, body)
    return body, answer


def main():
    rng = random.Random(SEED)
    df = load_pool1()
    for split, n in (("train", None), ("val", 300)):
        rows = df[df["split"] == split]
        if n:
            rows = rows.sample(n, random_state=SEED)
        with open(PROCESSED / f"numbers_{split}.jsonl", "w", encoding="utf-8") as fh:
            for _, row in rows.iterrows():
                note, answer = note_and_answer(row.to_dict(), rng)
                fh.write(json.dumps({"source": "task_numbers", "id": row["id"],
                                     "messages": [{"role": "system", "content": NUMBERS_SYSTEM},
                                                  {"role": "user", "content": f"Triage note:\n{note}"},
                                                  {"role": "assistant", "content": json.dumps(answer)}]}) + "\n")
        print(split, len(rows))
    print(open(PROCESSED / "numbers_train.jsonl", encoding="utf-8").readline()[:1500])


if __name__ == "__main__":
    main()
