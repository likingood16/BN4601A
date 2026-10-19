"""The vitals model as the app uses it: one call per patient, JSON-ready output.

    from vitals_model import VitalsModel, enforce_floor
    vm = VitalsModel()
    out = vm.assess(rr=24, spo2=89, sbp=118, hr=124, temp_c=38.4,
                    consciousness="alert", age=67, sex="male", pain_score=3)
    out["note"]            # one line for the main language model's prompt
    out["floor_priority"]  # least urgent priority the main model may give (1-5)
    final = enforce_floor(llm_priority, out["floor_priority"])

What it combines:
  * NEWS2 with the missing-data safety extension (news2.py)
  * p_urgent from the gradient-boosting classifier (notebook 03), i.e. the chance the
    experts would triage this patient KTAS 1-2
  * a safety floor: NEWS2 band floor, tightened to 3 when p_urgent >= threshold
"""
import json
import math
from pathlib import Path

import joblib
import pandas as pd

import ktas_data as kd
import news2
import vitals_classifier as vc

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
AVPU_TO_MENTAL = {"alert": 1, "confused": 2, "verbal": 2, "pain": 3, "unresponsive": 4}


def enforce_floor(llm_priority, floor_priority):
    """Safety rule: the final priority can be more urgent than the floor, never less."""
    return min(int(llm_priority), int(floor_priority))


class VitalsModel:
    def __init__(self, model_dir=MODEL_DIR, use_classifier_floor=True):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.model = joblib.load(model_dir / "urgent_model.joblib")
        self.threshold = self.meta["threshold"]
        self.floor_if_flagged = self.meta["floor_if_flagged"]
        self.use_classifier_floor = use_classifier_floor

    @staticmethod
    def _clean_inputs(vitals):
        """Fahrenheit -> Celsius, impossible values -> missing (with a warning)."""
        out, warnings = dict(vitals), []
        t = out.get("temp_c")
        if t is not None and not _nan(t) and t > 45:
            out["temp_c"] = round((t - 32) * 5 / 9, 1)
            warnings.append(f"temperature {t:g} read as °F -> {out['temp_c']} °C")
        for k, (lo, hi) in kd.PLAUSIBLE.items():
            v = out.get(k)
            if v is not None and not _nan(v) and not lo <= v <= hi:
                warnings.append(f"{k} = {v:g} is outside {lo}-{hi}; treated as not recorded")
                out[k] = None
        if (out.get("sbp") is not None and out.get("dbp") is not None
                and not _nan(out["sbp"]) and not _nan(out["dbp"]) and out["dbp"] >= out["sbp"]):
            warnings.append("diastolic >= systolic; both treated as not recorded")
            out["sbp"] = out["dbp"] = None
        return out, warnings

    def assess(self, rr=None, spo2=None, sbp=None, dbp=None, hr=None, temp_c=None,
               consciousness=None, on_oxygen=None, age=None, sex=None, pain_score=None,
               injury=None):
        """consciousness: alert / confused / verbal / pain / unresponsive.
        sex: 'male' / 'female'. injury: True if the visit is for an injury.
        Anything unknown can be left as None."""
        vitals, warnings = self._clean_inputs(
            {"rr": rr, "spo2": spo2, "sbp": sbp, "dbp": dbp, "hr": hr, "temp_c": temp_c,
             "nrs_pain": pain_score})
        cons = None if consciousness is None else str(consciousness).lower()
        if cons is not None and cons not in AVPU_TO_MENTAL:
            warnings.append(f"consciousness '{consciousness}' not recognised; treated as not recorded")
            cons = None

        res = news2.score(rr=vitals["rr"], spo2=vitals["spo2"], sbp=vitals["sbp"], hr=vitals["hr"],
                          temp_c=vitals["temp_c"], consciousness=cons, on_oxygen=on_oxygen)

        row = {
            "age": age, "is_male": None if sex is None else float(str(sex).lower().startswith("m")),
            "injured": None if injury is None else float(bool(injury)),
            "mental": AVPU_TO_MENTAL.get(cons), "nrs_pain": vitals["nrs_pain"],
            "sbp": vitals["sbp"], "dbp": vitals["dbp"], "hr": vitals["hr"], "rr": vitals["rr"],
            "temp_c": vitals["temp_c"], "spo2": vitals["spo2"], "news2": res["news2"],
        }
        X = pd.DataFrame([row])[vc.FEATURES].astype(float)
        p_urgent = float(vc.urgent_proba(self.model, X)[0])
        flagged = p_urgent >= self.threshold

        floor = res["floor_priority"]
        if self.use_classifier_floor and flagged:
            floor = min(floor, self.floor_if_flagged)

        note = news2.note(res)
        note += f" Vitals model: {p_urgent:.0%} chance of needing KTAS 1-2 care"
        note += f" (above the {self.threshold:.0%} alert threshold)." if flagged else "."
        if floor < 5:
            note += f" Safety floor: priority {floor} or more urgent."

        return {
            "news2": res["news2"],
            "news2_risk": res["risk_safe"],
            "subscores": res["subscores"],
            "escalated": res["escalated"],
            "escalation_reason": res["escalation_reason"],
            "p_urgent": round(p_urgent, 3),
            "urgent_flag": bool(flagged),
            "floor_priority": int(floor),
            "danger": floor <= 2,
            "missing": res["missing"],
            "assumed": res["assumed"],
            "warnings": warnings,
            "note": note,
        }


def _nan(x):
    return isinstance(x, float) and math.isnan(x)
