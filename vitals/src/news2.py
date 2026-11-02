"""NEWS2 (National Early Warning Score 2) calculator with explicit missing-data handling.

Bands follow the Royal College of Physicians NEWS2 chart (2017), SpO2 scale 1.

Standard NEWS2 assumes every parameter was measured. In the KTAS data, about half of
the most urgent patients (cardiac arrest, unresponsive) have no vitals at all, so
scoring a missing value as 0 would call them low risk. This module therefore returns:

  news2            standard score, missing parameters count as 0 (what a naive
                   calculator would report; kept for comparison)
  risk             standard NEWS2 risk band from that score
  risk_safe        same band, escalated to "high" when the patient could not be
                   measured (>= 2 of RR / SBP / HR missing) or is unresponsive
  floor_priority   the least urgent priority (1-5) the main model may assign

The escalation rule is a safety extension of NEWS2, not part of the official score.
"Unresponsive" follows triage scales themselves (KTAS/CTAS put an unresponsive
patient at level 1). The missing-vitals part was suggested by the cleaning audit,
which looked at all rows, so report it as a design choice informed by the data and
judge it on the held-out test split.
"""
import math

import pandas as pd

PARAMS = ["rr", "spo2", "on_oxygen", "sbp", "hr", "consciousness", "temp_c"]
NAMES = {"rr": "RR", "spo2": "SpO2", "on_oxygen": "O2", "sbp": "SBP", "hr": "HR",
         "consciousness": "Consciousness", "temp_c": "Temp"}
UNITS = {"rr": "/min", "spo2": "%", "sbp": " mmHg", "hr": " bpm", "temp_c": " °C"}
CORE = ["rr", "sbp", "hr"]          # missing >= 2 of these = "could not be measured"

# (upper bound inclusive, points), checked in order; last bound is infinity.
# Using upper bounds rather than ranges leaves no gaps for values like 35.05.
BANDS = {
    "rr":     [(8, 3), (11, 1), (20, 0), (24, 2), (math.inf, 3)],
    "spo2":   [(91, 3), (93, 2), (95, 1), (math.inf, 0)],
    "sbp":    [(90, 3), (100, 2), (110, 1), (219, 0), (math.inf, 3)],
    "hr":     [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (math.inf, 3)],
    "temp_c": [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (math.inf, 2)],
}
MENTAL_TO_AVPU = {1: "alert", 2: "verbal", 3: "pain", 4: "unresponsive"}

RISK_ORDER = ["low", "low-medium", "medium", "high"]
# Least urgent priority allowed for each band (1 = treat now ... 5 = can wait)
FLOOR = {"high": 2, "medium": 3, "low-medium": 3, "low": 5}


def _missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)


def band_points(param, value):
    for upper, pts in BANDS[param]:
        if value <= upper:
            return pts
    raise ValueError(f"{param}={value}")


def score(rr=None, spo2=None, sbp=None, hr=None, temp_c=None,
          consciousness=None, on_oxygen=None):
    """Score one patient. consciousness: 'alert', 'confused', 'verbal', 'pain',
    'unresponsive' (or None). on_oxygen: True / False / None (unknown -> assumed air).
    temp_c above 45 is taken to be Fahrenheit and converted."""
    vals = {"rr": rr, "spo2": spo2, "sbp": sbp, "hr": hr, "temp_c": temp_c}
    if not _missing(temp_c) and temp_c > 45:
        vals["temp_c"] = round((temp_c - 32) * 5 / 9, 1)

    sub, missing, assumed = {}, [], []
    for p, v in vals.items():
        if _missing(v):
            missing.append(p)
        else:
            sub[p] = band_points(p, float(v))

    if _missing(consciousness):
        missing.append("consciousness")
    else:
        sub["consciousness"] = 0 if str(consciousness).lower() == "alert" else 3

    if _missing(on_oxygen):
        assumed.append("on_oxygen")        # dataset has no oxygen field: assume room air
        sub["on_oxygen"] = 0
    else:
        sub["on_oxygen"] = 2 if on_oxygen else 0

    total = sum(sub.values())
    if total >= 7:
        risk = "high"
    elif total >= 5:
        risk = "medium"
    elif any(v == 3 for v in sub.values()):
        risk = "low-medium"
    else:
        risk = "low"

    unmeasurable = sum(p in missing for p in CORE) >= 2
    unresponsive = str(consciousness).lower() == "unresponsive"
    risk_safe = "high" if (unmeasurable or unresponsive) else risk
    reasons = []
    if unmeasurable:
        reasons.append("core vitals could not be recorded")
    if unresponsive:
        reasons.append("unresponsive")

    return {
        "news2": total,
        "subscores": sub,
        "risk": risk,
        "risk_safe": risk_safe,
        "escalated": risk_safe != risk,
        "escalation_reason": "; ".join(reasons),
        "missing": missing,
        "assumed": assumed,
        "floor_priority": FLOOR[risk_safe],
        "values": {**vals, "consciousness": consciousness},
    }


def note(result):
    """One-line text summary for the main language model."""
    v, sub = result["values"], result["subscores"]
    flagged = sorted((p for p in sub if sub[p] > 0 and p != "on_oxygen"),
                     key=lambda p: -sub[p])
    parts = []
    for p in flagged:
        if p == "consciousness":
            parts.append(f"responds to {v[p]}" if v[p] in ("verbal", "pain") else str(v[p]))
        else:
            parts.append(f"{NAMES[p]} {v[p]:g}{UNITS[p]} (+{sub[p]})")
    text = f"NEWS2 {result['news2']} = {result['risk_safe'].upper()} risk"
    if result["escalated"]:
        text += f" (escalated: {result['escalation_reason']})"
    text += ". " + ("Abnormal: " + ", ".join(parts) + "." if parts else "All recorded vitals in normal range.")
    if result["missing"]:
        text += " Not recorded: " + ", ".join(NAMES[p] for p in result["missing"]) + "."
    if "spo2" in result["missing"] and not result["escalated"]:
        text += " Score may be underestimated by up to 3."
    return text


def score_frame(df):
    """Score every row of the cleaned KTAS table. Returns a DataFrame aligned to df."""
    rows = []
    for r in df.itertuples(index=False):
        res = score(rr=r.rr, spo2=r.spo2, sbp=r.sbp, hr=r.hr, temp_c=r.temp_c,
                    consciousness=MENTAL_TO_AVPU.get(r.mental))   # None when not recorded
        rows.append({
            "news2": res["news2"],
            "risk": res["risk"],
            "risk_safe": res["risk_safe"],
            "escalated": res["escalated"],
            "floor_priority": res["floor_priority"],
            "n_missing": len(res["missing"]),
            **{f"sub_{p}": res["subscores"].get(p) for p in PARAMS if p != "on_oxygen"},
            "note": note(res),
        })
    out = pd.DataFrame(rows, index=df.index)
    for c in ["risk", "risk_safe"]:
        out[c] = pd.Categorical(out[c], categories=RISK_ORDER, ordered=True)
    return out
