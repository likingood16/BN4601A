"""Messy note -> numbers for the urgency model, each with the exact place in the note it came from.

Built on src/charting.extract_numbers (fixed rules, 100% of vital numbers right on the 190 KTAS test notes). Added here:
  - Prehospital values: a number in a sentence about the ambulance crew / EMS / before arrival ("EMS found SBP in the
    50s") is not an arrival observation. Those sentences are hidden first; a value found ONLY there is kept but marked
    "prehospital", so the nurse sees it and decides (real-AI check run 3 took the EMS heart rate as the arrival HR).
  - Spans (start, end) in the original note, so the screen can highlight the words each tile came from.
"""
import re

import charting

PREHOSPITAL = re.compile(
    r"\b(ems|emergency medical services|paramedics?|ambulance crew|pre-?hospital|en route|on scene|at the scene|"
    r"by the crew|scdf|"
    r"before arrival|prior to arrival|at home|gp (?:found|measured|noted)|at the clinic|polyclinic)\b", re.I)
SENTENCE = re.compile(r"[^.;\n]+(?:\.(?=\d)[^.;\n]*)*[.;\n]?")       # keeps "37.9" in one piece
VITALS = ["hr", "sbp", "dbp", "rr", "spo2", "temp_c", "nrs_pain", "mental"]
NAMES = {"age": "Age", "sex": "Sex", "hr": "Heart rate", "sbp": "BP systolic", "dbp": "BP diastolic",
         "rr": "Resp. rate", "spo2": "SpO₂", "temp_c": "Temperature", "nrs_pain": "Pain",
         "mental": "Consciousness", "ambulance": "Arrival", "injured": "Injury"}
UNITS = {"hr": "/min", "rr": "/min", "spo2": "%", "temp_c": "°C", "nrs_pain": "/10", "sbp": "mmHg", "dbp": "mmHg"}


def _mask(note):
    """The note with prehospital sentences blanked out (same length, so positions still line up)."""
    out = list(note)
    hidden = False
    for m in SENTENCE.finditer(note):
        if PREHOSPITAL.search(m.group(0)):
            hidden = True
            for i in range(m.start(), m.end()):
                if out[i] != "\n":
                    out[i] = " "
    return "".join(out), hidden


def _span(note, source, after=0):
    if not source:
        return None
    i = note.lower().find(source.lower(), after)
    return [i, i + len(source)] if i >= 0 else None


def extract(note):
    """-> {"fields": {...}, "source": {key: {"text", "span", "prehospital"}}, "warnings": [...]}"""
    masked, hidden = _mask(note)
    fields, info = charting.extract_numbers(masked)
    source = {k: {"text": v, "span": _span(note, v), "prehospital": False} for k, v in info["found"].items()}
    warnings = list(info["warnings"])
    if hidden:
        all_fields, all_info = charting.extract_numbers(note)
        for k, v in all_fields.items():
            if k in fields:
                continue
            src = all_info["found"].get(k)
            if k in VITALS:                             # only in the EMS sentence: shown, flagged, not used for scoring
                source[k] = {"text": src, "span": _span(note, src), "prehospital": True, "value": v}
                warnings.append(f"{NAMES.get(k, k)} {v:g} is from before arrival (“{src}”): not used. "
                                "Record the arrival value." if isinstance(v, (int, float)) else
                                f"{NAMES.get(k, k)} is from before arrival: not used.")
            else:                                       # age, sex, arrival mode are fine to take from anywhere
                fields[k] = v
                source[k] = {"text": src, "span": _span(note, src), "prehospital": False}
    s = source.get("spo2")
    if s and s.get("span"):                            # "sats 94% RA": the air/oxygen words come after the number
        after = note[s["span"][0]:s["span"][1] + 16].lower()
        s["air"] = ("room air" if re.search(r"\bra\b|room air|on air", after) else
                    "oxygen" if re.search(r"\bo2\b|oxygen|l/min|\bnc\b|nasal|mask|nrb", after) else None)
    return {"fields": fields, "source": source, "warnings": warnings}


def merge(old, new):
    """Numbers from a later note fill gaps and update vitals (a repeat set of obs is newer); age/sex never change."""
    fields, source = dict(old["fields"]), dict(old["source"])
    for k, v in new["fields"].items():
        if k in ("age", "sex") and k in fields:
            continue
        fields[k] = v
        source[k] = new["source"][k]
    for k, s in new["source"].items():
        if s.get("prehospital") and k not in source:
            source[k] = s
    return {"fields": fields, "source": source, "warnings": old.get("warnings", []) + new.get("warnings", [])}
