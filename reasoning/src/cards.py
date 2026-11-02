"""Patient cards: the one text format MedGemma reads, for every patient (same layout everywhere).

    from cards import make_card, build_pool1_cards
    card = make_card(row, vitals_out)          # one patient
    cards = build_pool1_cards()                # all Pool 1 patients -> processed/cards_pool1.parquet

The VITALS ASSESSMENT line comes from Caleb's VitalsModel. His saved model was refitted on ALL KTAS
patients (test included), so for honest test results we refit the same model on train + val only
(`dev_vitals_model`). Caleb's settings are reused unchanged (vitals/models/model_meta.json).
"""
import math

import numpy as np
import pandas as pd

from config import PROCESSED, SEED
from data import load_pool1, load_multicare
from ed_assessment import ed_assessment

AVPU_CODE = {"alert": 1, "verbal": 2, "pain": 3, "unresponsive": 4}
SPLIT_SEP = "=== ED PATIENT RECORD ==="


def _num(x, fmt="{:g}"):
    return "not recorded" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def dev_vitals_model():
    """Caleb's VitalsModel, but its urgency classifier refitted on KTAS train + val only."""
    import news2
    import vitals_classifier as vc
    from vitals_model import VitalsModel

    vm = VitalsModel()
    p = vm.meta["urgent_model"]["params"]
    dev = load_pool1(source="ktas")
    dev = dev[dev["split"] != "test"]
    X = pd.DataFrame({
        "age": dev["age"], "is_male": (dev["sex"] == "male").astype(float),
        "injured": dev["injured"], "mental": dev["mental"].map(AVPU_CODE).astype(float),
        "nrs_pain": dev["nrs_pain"], **{v: dev[v] for v in ["sbp", "dbp", "hr", "rr", "temp_c", "spo2"]},
    })
    X["news2"] = news2.score_frame(X.assign(mental=X["mental"]))["news2"].to_numpy()
    vm.model = vc.make_hgb(**p, seed=0).fit(X[vc.FEATURES].astype(float), dev["urgent"])
    vm.note_source = "VitalsModel refitted on KTAS train+val (no test patients)"
    return vm


def assess(vm, row):
    """Run Caleb's vitals model on one Pool 1 row."""
    v = lambda c: None if pd.isna(row.get(c)) else float(row[c])
    return vm.assess(rr=v("rr"), spo2=v("spo2"), sbp=v("sbp"), dbp=v("dbp"), hr=v("hr"), temp_c=v("temp_c"),
                     consciousness=None if pd.isna(row.get("mental")) else row["mental"],
                     age=v("age"), sex=row.get("sex"), pain_score=v("nrs_pain"),
                     injury=None if pd.isna(row.get("injured")) else bool(row["injured"]))


def make_card(row, vitals_out=None, imaging="none", notes="none", ed=False):
    """One patient card (the handoff's layout). Missing information says 'not recorded', never a guess.
    ed=True adds the ED ASSESSMENT section (ed_assessment.py) after the vitals line."""
    arrival = {1.0: "ambulance", 0.0: "walk-in / private transport"}.get(row.get("ambulance"), "not recorded")
    complaint = row.get("chief_complaint")
    complaint = complaint if isinstance(complaint, str) and complaint.strip() else "not recorded"
    if row.get("injured") == 1.0:
        complaint += " (injury)"
    bp = "not recorded" if pd.isna(row.get("sbp")) else f"{row['sbp']:g}/{_num(row.get('dbp'))}"
    obs = " | ".join([
        f"HR {_num(row.get('hr'))}", f"BP {bp}", f"RR {_num(row.get('rr'))}",
        f"SpO2 {_num(row.get('spo2'), '{:g}%')} (room air assumed)", f"Temp {_num(row.get('temp_c'), '{:.1f} C')}",
        f"Consciousness {row.get('mental') if isinstance(row.get('mental'), str) else 'not recorded'}",
        f"Pain {_num(row.get('nrs_pain'), '{:g}/10')}",
    ])
    lines = [
        SPLIT_SEP,
        f"PATIENT: {_num(row.get('age'), '{:.0f}')} {row.get('sex') or 'sex not recorded'} | Arrival: {arrival} | Time: not recorded",
        f"PRESENTING COMPLAINT: {complaint}",
        "HISTORY OF PRESENTING COMPLAINT: not recorded",
        "PAST HISTORY: not recorded",
        "MEDICATIONS: not recorded",
        "ALLERGIES: not recorded",
        f"OBSERVATIONS: {obs}",
        f"VITALS ASSESSMENT: {vitals_out['note'] if vitals_out else 'not available'}",
        *([ed_assessment(row)] if ed else []),
        f"IMAGING: {imaging}",
        f"CONSULTATION NOTES: {notes}",
        "=== END RECORD ===",
    ]
    return "\n".join(lines)


def build_pool1_cards(limit=None, save=True, ed=False):
    """Cards + Caleb's floor/p_urgent for every Pool 1 patient. Saved for the embedding/heads jobs.
    ed=True -> cards with the ED ASSESSMENT section, saved as cards_pool1_ed.parquet (the original stays)."""
    vm = dev_vitals_model()
    df = load_pool1()
    if limit:
        df = df.groupby(["source", "split"]).head(limit)
    rows = []
    for _, r in df.iterrows():
        out = assess(vm, r)
        rows.append({"id": r["id"], "source": r["source"], "split": r["split"],
                     "triage_level": r["triage_level"], "urgent": r["urgent"],
                     "card": make_card(r, out, ed=ed), "news2": out["news2"], "vitals_p_urgent": out["p_urgent"],
                     "floor_priority": out["floor_priority"], "danger": out["danger"]})
    cards = pd.DataFrame(rows)
    if save:
        cards.to_parquet(PROCESSED / ("cards_pool1_ed.parquet" if ed else "cards_pool1.parquet"), index=False)
    return cards


def make_case_card(case):
    """MultiCaRe case report as a card: the narrative goes in, nothing is pre-extracted."""
    text = " ".join(str(case["case_text"]).split())
    return "\n".join([
        SPLIT_SEP,
        f"PATIENT: {case['age']:.0f} {str(case['gender']).lower()} | Arrival: emergency department | Time: not recorded",
        f"CASE NARRATIVE (from published case report): {text[:3000]}",
        "IMAGING: see narrative",
        "CONSULTATION NOTES: none",
        "=== END RECORD ===",
    ])


def build_multicare_cards(save=True):
    mc = load_multicare()
    mc["card"] = [make_case_card(c) for _, c in mc.iterrows()]
    out = mc[["case_id", "department_guess", "card"]]
    if save:
        out.to_parquet(PROCESSED / "cards_multicare.parquet", index=False)
    return out
