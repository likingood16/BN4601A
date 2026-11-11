"""ED ASSESSMENT section for the patient card: the structured first look an ED clinician does (ABCDE, quick screens,
pathway flags), built ONLY from fields that exist in the record. Missing data says "not recorded", never a guess.

    from ed_assessment import ed_assessment
    text = ed_assessment(row)            # row = one Pool 1 patient (pool1.parquet columns)

Sources for the cut-offs (checked 29 Oct 2026):
  - A/B/C/E bands: NEWS2 (Royal College of Physicians 2017), using Caleb's own scoring (vitals/src/news2.py),
    so the card agrees with his VITALS ASSESSMENT line: 0 points = normal, 1-2 = abnormal, 3 = critical.
  - Shock index = HR / SBP; >= 1 marks possible shock (about 10x odds of 30-day death in ~111k ED visits,
    PMC4470645).
  - qSOFA (Sepsis-3, Seymour et al. JAMA 2016): RR >= 22, SBP <= 100, altered mentation; >= 2 = higher risk.
  - Chest pain: ECG within 10 minutes of arrival (AHA/ACC 2021 chest pain guideline, Class 1).
  - Possible stroke: time last known well decides acute treatment (AHA/ASA acute stroke guideline).
"""
import math
import re

import news2

LEVEL = {0: "normal", 1: "abnormal", 2: "abnormal", 3: "CRITICAL"}
PATHWAYS = [   # (words in the presenting complaint, flag)
    (r"chest (pain|tight|discomfort|pressure)|angina", "chest pain -> ECG within 10 minutes of arrival"),
    (r"weakness|numb|slurr|speech|aphasi|dysarthri|facial (droop|palsy)|hemipleg|paralys|stroke|\bcva\b",
     "possible stroke -> record time last known well, glucose, urgent brain imaging"),
    (r"dyspn|short(ness)? of breath|breath|respirat|wheez", "breathing complaint -> oxygen saturation and work of breathing"),
    (r"syncop|faint|collapse|loss of consciousness", "collapse/syncope -> ECG, glucose, check for injury"),
    (r"seiz|convuls|epilep", "seizure -> glucose, airway protection, time of seizure"),
    (r"overdose|poison|intoxic|ingest", "possible poisoning -> find out the substance, amount and time"),
    (r"fever|infect|sepsis|chills", "possible infection -> sepsis screen (qSOFA)"),
]


def _missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _val(row, col):
    v = row.get(col)
    return None if _missing(v) else v


def _band(sub, param, text):
    return f"{text} ({LEVEL[sub[param]]})" if param in sub else f"{text.split()[0]} not recorded"


def ed_assessment(row, flags=None):
    """flags: ED pathway flags from the trained heads (pathway_heads.py). None = the old word rules (test mode)."""
    rr, spo2, sbp, hr, temp = (_val(row, c) for c in ("rr", "spo2", "sbp", "hr", "temp_c"))
    mental, pain, injured = _val(row, "mental"), _val(row, "nrs_pain"), _val(row, "injured")
    n2 = news2.score(rr=rr, spo2=spo2, sbp=sbp, hr=hr, temp_c=temp, consciousness=mental)
    sub = n2["subscores"]
    fmt = lambda x: f"{x:g}"

    breathing = "; ".join([_band(sub, "rr", f"RR {fmt(rr)}" if rr is not None else "RR"),
                           _band(sub, "spo2", f"SpO2 {fmt(spo2)}%" if spo2 is not None else "SpO2")])
    si = hr / sbp if hr is not None and sbp else None
    shock = ("shock index not recorded" if si is None else
             f"shock index {si:.2f}" + (" (>= 1: possible shock)" if si >= 1 else " (normal)"))
    circulation = "; ".join([_band(sub, "hr", f"HR {fmt(hr)}" if hr is not None else "HR"),
                             _band(sub, "sbp", f"SBP {fmt(sbp)}" if sbp is not None else "SBP"), shock])
    disability = "; ".join([
        "consciousness not recorded" if mental is None else
        f"consciousness {mental}" + ("" if str(mental).lower() == "alert" else " (ALTERED)"),
        "pain not recorded" if pain is None else
        f"pain {fmt(pain)}/10" + (" (severe)" if pain >= 7 else " (moderate)" if pain >= 4 else "")])
    exposure = "; ".join([_band(sub, "temp_c", f"temp {temp:.1f} C" if temp is not None else "temp"),
                          "injury not recorded" if injured is None else ("injury: yes" if injured else "injury: no")])

    # qSOFA: count the criteria we can see, and say which could not be checked
    checks = {"RR >= 22": None if rr is None else rr >= 22, "SBP <= 100": None if sbp is None else sbp <= 100,
              "altered mentation": None if mental is None else str(mental).lower() != "alert"}
    met = [k for k, v in checks.items() if v]
    unknown = [k for k, v in checks.items() if v is None]
    qsofa = f"qSOFA {len(met)}/3" + (f" ({', '.join(met)})" if met else "")
    if unknown:
        qsofa += f"; not recorded: {', '.join(unknown)}"
    if len(met) >= 2:
        qsofa += " -> higher sepsis risk if infection suspected"

    from_ai = flags is not None
    if flags is None:      # test mode and old training cards only: the word rules (the app uses trained heads)
        complaint = str(row.get("chief_complaint") or "").lower()
        flags = [flag for pattern, flag in PATHWAYS if re.search(pattern, complaint)]
    flags = list(flags)
    if injured:
        # "not recorded" was printed even when the note said "fell off ladder" (10 Nov, B3): this file sees only
        # the numbers, so it no longer claims that; charting.pathway_todos checks the note's own words
        flags.append("trauma -> mechanism of injury; primary survey")
    if str(mental).lower() == "unresponsive":
        flags.insert(0, "UNRESPONSIVE -> immediate resuscitation assessment")

    return "\n".join([
        "ED ASSESSMENT (NEWS2 bands from the recorded vitals; pathway flags from trained AI heads):" if from_ai else "ED ASSESSMENT (rules applied to the recorded data):",
        f"  A/B airway-breathing: {breathing}",
        f"  C circulation: {circulation}",
        f"  D disability: {disability}",
        f"  E exposure: {exposure}",
        f"  Screens: {qsofa}",
        f"  Pathway flags: {'; '.join(flags) if flags else 'none'}",
    ])
