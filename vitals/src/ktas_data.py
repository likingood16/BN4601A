"""Load, clean and validate the Kaggle KTAS emergency triage dataset.

Source: kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application
Original study: Moon S-H, Shim JL, Park K-S, Park C-S (2019). Triage accuracy and
causes of mistriage using the Korean Triage and Acuity Scale. PLoS One 14(9):e0216972.

Raw-file quirks handled here (found by auditing the CSV):
  * semicolon separated, Turkish Windows encoding (cp1254), decimal commas
  * missing vitals written as '??'; missing pain scores written as '#BOŞ!'
    (Turkish Excel's '#EMPTY!')
  * trailing spaces ('0 ') and trailing dots ('36.') in numeric fields
  * 5 visits recorded twice, identical except for a second ED diagnosis
  * one implausible SpO2 of 20% in an alert patient triaged KTAS 3

Missing vitals are NOT random: about half of KTAS 1 patients have no vitals
(cardiac arrest, unresponsive) versus <1% for KTAS 3-4. Downstream scoring must
never treat a missing vital as normal.

Usage:
    raw = load_raw(RAW_PATH)
    df, log = clean(raw)
    report = validate(df)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ktas_triage.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

PLACEHOLDERS = ["??", "#BOŞ!"]

RENAME = {
    "Group": "ed_type",
    "Sex": "sex",
    "Age": "age",
    "Patients number per hour": "patients_per_hour",
    "Arrival mode": "arrival_mode",
    "Injury": "injury",
    "Chief_complain": "chief_complaint",
    "Mental": "mental",
    "Pain": "pain",
    "NRS_pain": "nrs_pain",
    "SBP": "sbp",
    "DBP": "dbp",
    "HR": "hr",
    "RR": "rr",
    "BT": "temp_c",
    "Saturation": "spo2",
    "KTAS_RN": "ktas_nurse",
    "Diagnosis in ED": "diagnosis",
    "Disposition": "disposition",
    "KTAS_expert": "ktas_expert",
    "Error_group": "error_group",
    "Length of stay_min": "los_min",
    "KTAS duration_min": "ktas_duration_min",
    "mistriage": "mistriage",
}
TEXT_COLS = ["chief_complaint", "diagnosis"]

# Code books from the Kaggle data card (spelling fixed). Disposition 1 and 4 are
# both listed as "discharge" in the source, so 4 is kept as its own label.
CODES = {
    "ed_type": {1: "local_ed_3rd_degree", 2: "regional_ed_4th_degree"},
    "sex": {1: "female", 2: "male"},
    "injury": {1: "no", 2: "yes"},
    "pain": {0: "no", 1: "yes"},
    "mental": {1: "alert", 2: "verbal", 3: "pain", 4: "unresponsive"},
    "arrival_mode": {1: "walking", 2: "public_ambulance", 3: "private_vehicle",
                     4: "private_ambulance", 5: "other", 6: "other", 7: "other"},
    "disposition": {1: "discharge", 2: "ward", 3: "icu", 4: "discharge_code4",
                    5: "transfer", 6: "death", 7: "surgery"},
    # Checked against the data: 1 = nurse more urgent than experts, 2 = less urgent
    "mistriage": {0: "correct", 1: "over_triage", 2: "under_triage"},
}

VITALS = ["sbp", "dbp", "hr", "rr", "temp_c", "spo2"]

# Values outside these limits are treated as recording errors and set to missing.
# Wide on purpose: genuine extremes (SBP 50, HR 32) do occur in KTAS 1-2 patients.
PLAUSIBLE = {
    "sbp": (40, 300),
    "dbp": (20, 200),
    "hr": (20, 250),
    "rr": (4, 60),
    "temp_c": (30.0, 43.0),
    "spo2": (50, 100),
    "nrs_pain": (0, 10),
}

# Known at the triage desk -> usable as model inputs
TRIAGE_FEATURES = [
    "age", "is_male", "arrival_mode", "injured", "chief_complaint", "mental",
    "has_pain", "nrs_pain", *VITALS, "ed_type", "patients_per_hour",
]
# Known only after triage, or another version of the answer -> never a model input
LEAKAGE_COLS = [
    "ktas_nurse", "diagnosis", "disposition", "error_group",
    "los_min", "ktas_duration_min", "mistriage",
]
TARGET = "ktas_expert"


def load_raw(path=RAW_PATH):
    """Read the CSV exactly as stored: every column as text, nothing coerced."""
    return pd.read_csv(path, sep=";", encoding="cp1254", dtype=str, keep_default_na=False)


def clean(raw):
    """Return (clean DataFrame, cleaning log DataFrame)."""
    log = []

    def note(step, column, n, detail):
        log.append({"step": step, "column": column, "n_rows": int(n), "detail": detail})

    df = raw.rename(columns=RENAME).copy()
    note("rename", "all", len(df), "columns renamed to snake_case")

    for c in df.columns:
        df[c] = df[c].str.strip()
        for p in PLACEHOLDERS:
            n = (df[c] == p).sum()
            if n:
                note("placeholder -> missing", c, n, f"'{p}'")
        df[c] = df[c].replace(PLACEHOLDERS + [""], np.nan)

    for c in df.columns.difference(TEXT_COLS):
        n_dot = df[c].str.endswith(".", na=False).sum()
        if n_dot:
            note("trailing dot", c, n_dot, "e.g. '36.' read as 36.0")
        df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="raise")

    # Merge visits that were recorded once per diagnosis
    triage_cols = [c for c in RENAME.values() if c not in LEAKAGE_COLS]
    dup = df.duplicated(triage_cols, keep=False)
    if dup.any():
        n_before = len(df)
        agg = {c: "first" for c in df.columns if c not in triage_cols}
        agg["diagnosis"] = lambda s: " | ".join(s.dropna().unique())
        df = (df.reset_index()
                .groupby(triage_cols, dropna=False, sort=False)
                .agg({"index": "first", **agg})
                .reset_index()
                .sort_values("index")
                .drop(columns="index")
                .reset_index(drop=True)[list(RENAME.values())])
        note("merge duplicate visits", "diagnosis", n_before - len(df),
             f"{dup.sum()} rows had identical triage data and differed only in diagnosis; "
             "diagnoses joined with ' | '")

    for c, (lo, hi) in PLAUSIBLE.items():
        bad = df[c].notna() & ~df[c].between(lo, hi)
        if bad.any():
            note("implausible -> missing", c, bad.sum(),
                 f"outside {lo}-{hi}: {sorted(df.loc[bad, c].tolist())}")
            df.loc[bad, c] = np.nan
    bad_bp = df["dbp"] >= df["sbp"]
    if bad_bp.any():
        note("implausible -> missing", "sbp,dbp", bad_bp.sum(), "DBP >= SBP")
        df.loc[bad_bp, ["sbp", "dbp"]] = np.nan

    # No pain reported -> pain score is 0, not unknown
    fill = (df["pain"] == 0) & df["nrs_pain"].isna()
    df.loc[fill, "nrs_pain"] = 0
    note("logical fill", "nrs_pain", fill.sum(), "pain = no -> NRS 0")
    left = df["nrs_pain"].isna().sum()
    if left:
        note("left missing", "nrs_pain", left, "pain = yes but no score recorded")

    df["is_male"] = (df["sex"] == 2).astype(int)
    df["injured"] = (df["injury"] == 2).astype(int)
    df["has_pain"] = df["pain"].astype(int)
    for c, book in CODES.items():
        df[f"{c}_label"] = df[c].map(book)

    for v in VITALS:
        df[f"{v}_missing"] = df[v].isna().astype(int)
    df["n_vitals_missing"] = df[[f"{v}_missing" for v in VITALS]].sum(axis=1)
    df["age_under_18"] = (df["age"] < 18).astype(int)
    note("flag", "age", df["age_under_18"].sum(), "aged 16-17 (study counted >=15 as adult); kept")

    int_cols = ["ed_type", "sex", "age", "patients_per_hour", "arrival_mode", "injury",
                "mental", "pain", "ktas_nurse", "disposition", "ktas_expert",
                "error_group", "los_min", "mistriage"]
    df[int_cols] = df[int_cols].astype(int)
    df.insert(0, "visit_id", np.arange(1, len(df) + 1))
    return df, pd.DataFrame(log)


def validate(df):
    """Run checks on the cleaned data. Returns one row per check."""
    checks = []

    def check(name, failing, detail="", severity="error"):
        failing = int(failing)
        checks.append({"check": name, "passed": failing == 0, "n_failing": failing,
                       "severity": severity, "detail": detail})

    check("one row per visit (no repeated triage records)",
          df.duplicated([c for c in RENAME.values() if c not in LEAKAGE_COLS]).sum())
    check("visit_id unique", df["visit_id"].duplicated().sum())
    missing_cols = set(TRIAGE_FEATURES + LEAKAGE_COLS + [TARGET]) - set(df.columns)
    check("all expected columns present", len(missing_cols), ", ".join(sorted(missing_cols)))

    for c in ["ktas_expert", "ktas_nurse"]:
        check(f"{c} in 1-5 and never missing", (~df[c].isin(range(1, 6))).sum())
    for c, book in CODES.items():
        check(f"{c} codes valid", (~df[c].isin(book)).sum(), f"allowed {sorted(book)}")

    for c, (lo, hi) in PLAUSIBLE.items():
        check(f"{c} within {lo}-{hi} or missing", (df[c].notna() & ~df[c].between(lo, hi)).sum())
    check("dbp < sbp when both present", (df["dbp"] >= df["sbp"]).sum())
    check("no pain -> nrs_pain is 0", ((df["has_pain"] == 0) & (df["nrs_pain"] != 0)).sum())

    expected = np.select([df["ktas_nurse"] == df["ktas_expert"], df["ktas_nurse"] < df["ktas_expert"]],
                         [0, 1], 2)
    check("mistriage matches nurse vs expert KTAS", (df["mistriage"] != expected).sum())

    text = df.select_dtypes(include=["object", "string"])
    left = sum(text[c].isin(PLACEHOLDERS).sum() for c in text.columns)
    check("no placeholder strings left", left, "'??', '#BOŞ!'")
    check("age >= 15", (df["age"] < 15).sum())
    check("no leakage column in TRIAGE_FEATURES", len(set(TRIAGE_FEATURES) & set(LEAKAGE_COLS)))
    check("chief complaint recorded", df["chief_complaint"].isna().sum(),
          "'??' in the raw file; rows kept, complaint left missing", severity="warning")

    return pd.DataFrame(checks)


def missingness(df):
    """Missing count and % per vital, overall and by expert KTAS level."""
    out = pd.DataFrame({
        "missing": df[VITALS + ["nrs_pain"]].isna().sum(),
        "missing_%": df[VITALS + ["nrs_pain"]].isna().mean().mul(100).round(1),
    })
    by_ktas = df.groupby(TARGET)[VITALS].apply(lambda g: g.isna().mean().mul(100).round(1)).T
    by_ktas.columns = [f"KTAS {k} %" for k in by_ktas.columns]
    return out.join(by_ktas)


def make_splits(df, test_size=0.15, val_size=0.15, seed=42):
    """Stratified train/val/test split by expert KTAS. Returns visit_id -> split."""
    from sklearn.model_selection import train_test_split

    ids, y = df["visit_id"], df[TARGET]
    rest_ids, test_ids = train_test_split(ids, test_size=test_size, stratify=y, random_state=seed)
    rest_y = y[ids.isin(rest_ids)]
    train_ids, val_ids = train_test_split(rest_ids, test_size=val_size / (1 - test_size),
                                          stratify=rest_y, random_state=seed)
    split = pd.Series("train", index=ids.values, name="split")
    split[val_ids.values] = "val"
    split[test_ids.values] = "test"
    return split.rename_axis("visit_id").reset_index()


def load_clean(out_dir=PROCESSED_DIR):
    """Read the saved cleaned data, with its split column if splits.csv exists."""
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "ktas_clean.csv", encoding="utf-8-sig")
    if (out_dir / "splits.csv").exists():
        df = df.merge(pd.read_csv(out_dir / "splits.csv"), on="visit_id", validate="one_to_one")
    return df


def save(df, log, out_dir=PROCESSED_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ktas_clean.csv", index=False, encoding="utf-8-sig")
    log.to_csv(out_dir / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    return out_dir
