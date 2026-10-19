"""Map the NHAMCS 2022 emergency department public-use file onto the KTAS feature table,
for external validation of the vitals model.

Source: CDC National Center for Health Statistics, National Hospital Ambulatory Medical
Care Survey (NHAMCS) 2022 ED public-use file, Stata version (ed2022-stata.dta), codebook
doc22-ed-508.pdf. US government data, public domain.

Coding decisions (from the value labels in ed2022.do):
  * label: IMMEDR 1-5 (immediate / emergent / urgent / semi-urgent / nonurgent), assigned
    by the triage nurse with the ED's own system. 0 and 7 (no triage) and -8/-9 dropped.
    Urgent = 1-2, matching KTAS 1-2 in the training data.
  * TEMPF is °F x 10 (986 = 98.6 °F); 998 there is a real value (99.8 °F)
  * PULSE 998 and BPDIAS 998 mean "Doppler / palpated" (no number) -> missing
  * all negative codes (-9 blank, -8 unknown) -> missing
  * INJURY 0 = no, 1 = yes, 2 = questionable -> missing
  * SEX 1 = female, 2 = male (same coding as KTAS)
  * no consciousness (AVPU) field exists -> mental is missing for everyone
  * adults only: AGE >= 16, the youngest age in the KTAS training data
"""
from pathlib import Path

import numpy as np
import pandas as pd

import ktas_data as kd

RAW = Path(__file__).resolve().parents[1] / "data" / "external" / "nhamcs_2022" / "raw" / "ed2022-stata.dta"
COLS = ["AGE", "SEX", "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE",
        "INJURY", "ARREMS", "IMMEDR", "PATWT"]


def load(path=RAW):
    return pd.read_stata(path, columns=COLS, convert_categoricals=False)


def harmonize(raw):
    """Return (table in KTAS column names, exclusion log)."""
    log = [{"step": "NHAMCS 2022 ED visits", "n": len(raw)}]
    d = raw[raw.AGE >= 16]
    log.append({"step": "aged 16 or over", "n": len(d)})
    d = d[d.IMMEDR.between(1, 5)]
    log.append({"step": "triage level 1-5 recorded", "n": len(d)})

    neg = lambda s: s.where(s >= 0).astype(float)
    out = pd.DataFrame(index=d.index)
    out["age"] = d.AGE.astype(float)
    out["is_male"] = (d.SEX == 2).astype(float)
    out["injured"] = d.INJURY.map({0: 0.0, 1: 1.0})
    out["mental"] = np.nan
    out["nrs_pain"] = neg(d.PAINSCALE)
    out["sbp"] = neg(d.BPSYS)
    out["dbp"] = neg(d.BPDIAS).replace(998, np.nan)
    out["hr"] = neg(d.PULSE).replace(998, np.nan)
    out["rr"] = neg(d.RESPR)
    out["temp_c"] = ((neg(d.TEMPF) / 10 - 32) * 5 / 9).round(1)
    out["spo2"] = neg(d.POPCT)
    out["ambulance"] = d.ARREMS.map({1: 1.0, 2: 0.0})
    out["triage_level"] = d.IMMEDR.astype(int)
    out["urgent"] = (d.IMMEDR <= 2).astype(int)
    out["weight"] = d.PATWT.astype(float)

    # Same plausibility limits as the KTAS cleaning
    n_bad = 0
    for c, (lo, hi) in kd.PLAUSIBLE.items():
        bad = out[c].notna() & ~out[c].between(lo, hi)
        n_bad += int(bad.sum())
        out.loc[bad, c] = np.nan
    bad_bp = out.dbp >= out.sbp
    out.loc[bad_bp, ["sbp", "dbp"]] = np.nan
    log.append({"step": f"implausible values set to missing: {n_bad + 2 * int(bad_bp.sum())} values", "n": len(out)})
    return out.reset_index(drop=True), pd.DataFrame(log)
