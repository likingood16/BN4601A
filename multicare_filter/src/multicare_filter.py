"""Select the MultiCaRe images and cases relevant to the A&E triage project.

Data: OpenMed/multicare-images (161,702 images) joined to OpenMed/multicare-cases on
images.patient_id == cases.case_id. Uses the local Hugging Face cache if the datasets are
already there, otherwise downloads them once (~3.3 GB, e.g. on Google Colab). Image bytes are
only loaded for the few images shown in the notebook.

Every rule is one step in FILTERS, so the funnel shows what each rule removes and any rule
can be switched off or changed in CONFIG. Nothing here touches the vitals/ folder.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def _dataset_dir(repo_id, patterns):
    """Local folder of a Hugging Face dataset: cached copy if present, else download it."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo_id, repo_type="dataset", allow_patterns=patterns, local_files_only=True)
    except Exception:
        print(f"Downloading {repo_id} from Hugging Face (first run only)...")
        return snapshot_download(repo_id, repo_type="dataset", allow_patterns=patterns)


def _image_files():
    files = sorted(glob.glob(os.path.join(_dataset_dir("OpenMed/multicare-images", ["data/*.parquet"]),
                                          "data", "*.parquet")))
    if not files:
        raise FileNotFoundError("no image parquet files found for OpenMed/multicare-images")
    return files


def _case_file():
    return os.path.join(_dataset_dir("OpenMed/multicare-cases", ["cases.parquet"]), "cases.parquet")

CONFIG = {
    # Reuse allowed for a non-commercial student project; ND (no derivatives) and unlicensed excluded
    "licenses": ["CC0", "CC BY", "CC BY-SA", "CC BY-NC", "CC BY-NC-SA"],
    # A&E imaging; MRI, PET etc. are rarely first-line in A&E
    "radiology_subtypes": ["x_ray", "ct", "ultrasound"],
    # Clinical photographs kept only when the caption describes an acute A&E problem
    "photo_subtypes": ["skin_photograph", "other_medical_photograph"],
    "include_ecg": False,           # ECGs are A&E-relevant but hard for a VLM; switch on if wanted
    "min_caption_chars": 20,
    "exclude_later_timepoints": True,   # drop post-operative / follow-up / healed images
    "exclude_echo": False,          # echocardiograms are many of the ultrasound images; flagged either way
    "min_age": 16,
    "require_ed": True,
    "min_vitals": 1,                # measured vital signs mentioned in the case text
}

ACUTE_PHOTO = re.compile(
    r"wound|lacerat|burn|abscess|cellulitis|necroti[sz]|gangren|trauma|injur|bite|fractur|"
    r"dislocat|deformit|ha?ematoma|bruis|ecchymo|swelling|swollen|abrasion|stab|gunshot|"
    r"degloving|amputat|foreign body|impale|crush|blister|erythema|purpura|petechi|rash",
    re.IGNORECASE)
# Photos taken in theatre or of specimens are not what the A&E doctor sees on arrival
NOT_PRESENTATION_PHOTO = re.compile(
    r"intra-?operative|post-?operative|operative (view|field|finding|image)|surgical (field|specimen)|"
    r"specimen|resected|after (surgery|repair|debridement)|during (surgery|the operation)",
    re.IGNORECASE)

# Images from later in the stay (after surgery, at follow-up) are not what triage sees
LATER_TIMEPOINT = re.compile(
    r"post-?operative|follow-?up|after (treatment|therapy|surgery|repair|drainage|antibiotic|operation|"
    r"debridement|intervention|stenting|embolization|resection|discharge)|\d+\s*(months?|weeks?|years?) later|"
    r"\bheal(ed|ing)\b|resolution|resolved|at discharge|on discharge|postoperative day|"
    r"day \d+ (post|after)|\d+(st|nd|rd|th) (postoperative|post-operative) day",
    re.IGNORECASE)
ECHO = re.compile(r"echocardiogra|\bTTE\b|\bTEE\b|doppler", re.IGNORECASE)

ED_WORDS = re.compile(
    r"emergency (department|room|unit|ward|centre|center|service)|accident (and|&) emergency"
    r"|\bA&E\b|casualty department|trauma (bay|centre|center|unit)", re.IGNORECASE)
ED_ABBREV = re.compile(r"\b(to|in|at|from|into) (the |our )?(ED|ER)\b")

DEG = r"\s*(?:°|º|˚|℃|degrees?)\s*"
VITALS = {
    "blood_pressure": r"\b\d{2,3}\s*/\s*\d{2,3}\s*mm\s?hg|blood pressure\D{0,20}\d{2,3}",
    "heart_rate": r"(heart rate|pulse rate|pulse|\bHR\b)\D{0,15}\d{2,3}|\d{2,3}\s*(beats per minute|beats/min|bpm)",
    "resp_rate": r"(respiratory rate|respiration rate|\bRR\b)\D{0,15}\d{1,2}|\d{1,2}\s*breaths",
    "oxygen_sat": r"(oxygen saturation|SpO2|SaO2|O2 sat\w*|saturation)\D{0,25}\d{2,3}\s*%",
    "temperature": rf"\b\d{{2,3}}(\.\d)?{DEG}[CF]\b|\b\d{{2,3}}(\.\d)?\s*℃|temperature\D{{0,15}}\d{{2,3}}",
}
_nocap = lambda p: re.sub(r"\((?!\?)", "(?:", p)    # non-capturing groups: str.contains only needs a match
VITAL_RES = {k: re.compile(_nocap(v), re.IGNORECASE) for k, v in VITALS.items()}

# Rough department label from the case text, for hand-checking (the dataset has no department column).
DEPARTMENTS = {
    "Cardiology": r"cardiolog|myocardial infarction|\bSTEMI\b|NSTEMI|acute coronary|angina|"
                  r"cardiac tamponade|pericardial effusion|aortic dissection|heart failure|arrhythmi|"
                  r"atrial fibrillation|ventricular tachycardia|endocarditis|cardiogenic",
    "Respiratory": r"pneumonia|pulmonary embol|pneumothorax|pleural effusion|empyema|asthma|COPD|"
                   r"respiratory failure|pulmonolog|respiratory medicine|hemoptysis|haemoptysis|"
                   r"bronch|ARDS|lung abscess",
    "Orthopaedics": r"orthop(a)?edic|fracture|dislocation|osteomyelitis|septic arthritis|"
                    r"compartment syndrome|tendon rupture|spinal fracture",
    "Surgery": r"general surg|appendicitis|bowel obstruction|intestinal obstruction|perforat|"
               r"laparotomy|laparoscop|cholecystitis|volvulus|hernia|peritonitis|ischemic bowel|"
               r"ischaemic bowel|pancreatitis|abscess drainage|intussusception",
    "Neurology": r"neurolog|neurosurg|stroke|cerebral infarct|intracranial|subdural|subarachnoid|"
                 r"epidural ha?ematoma|seizure|status epilepticus|meningitis|encephalitis|"
                 r"cerebral venous|hydrocephalus|spinal cord compression",
}
DEPT_RES = {k: re.compile(_nocap(v), re.IGNORECASE) for k, v in DEPARTMENTS.items()}
ACUTE_PHOTO, NOT_PRESENTATION_PHOTO, LATER_TIMEPOINT, ECHO, ED_WORDS, ED_ABBREV = (
    re.compile(_nocap(r.pattern), r.flags)
    for r in (ACUTE_PHOTO, NOT_PRESENTATION_PHOTO, LATER_TIMEPOINT, ECHO, ED_WORDS, ED_ABBREV))


def load_images():
    files = _image_files()
    cols = [c for c in pq.read_schema(files[0]).names if c != "image"]
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)


def load_cases():
    rows = pq.read_table(_case_file()).to_pylist()
    cases = pd.DataFrame([{**c, "article_id": a["article_id"]} for a in rows for c in a["cases"]])
    cases["age"] = pd.to_numeric(cases["age"], errors="coerce")
    return cases


def annotate_cases(cases):
    text = cases["case_text"].fillna("")
    out = cases.copy()
    out["ed_presentation"] = text.str.contains(ED_WORDS) | text.str.contains(ED_ABBREV)
    for k, rx in VITAL_RES.items():
        out[f"v_{k}"] = text.str.contains(rx)
    out["n_vitals"] = out[[f"v_{k}" for k in VITALS]].sum(axis=1)
    counts = pd.DataFrame({d: text.str.count(rx) for d, rx in DEPT_RES.items()})
    best = counts.max(axis=1)
    out["department_guess"] = np.where(best == 0, "Other/unclear", counts.idxmax(axis=1))
    # Two departments tied at the top -> flag for hand-checking
    out["department_tied"] = (counts.eq(best, axis=0).sum(axis=1) > 1) & (best > 0)
    return out


def apply_filters(img, cases, cfg=CONFIG):
    """Return (selected images with case info, funnel table)."""
    df = img.merge(cases, left_on="patient_id", right_on="case_id", how="left")
    caption = df["caption"].fillna("")

    is_rad = (df.image_type == "radiology") & df.image_subtype.isin(cfg["radiology_subtypes"])
    is_photo = (df.image_type == "medical_photograph") & df.image_subtype.isin(cfg["photo_subtypes"]) \
        & caption.str.contains(ACUTE_PHOTO) & ~caption.str.contains(NOT_PRESENTATION_PHOTO)
    is_ecg = (df.image_subtype == "ekg") if cfg["include_ecg"] else pd.Series(False, index=df.index)

    steps = [
        ("all images", pd.Series(True, index=df.index)),
        ("licence allows reuse", df.license.isin(cfg["licenses"])),
        ("A&E modality (X-ray/CT/US, acute photo" + (", ECG" if cfg["include_ecg"] else "") + ")",
         is_rad | is_photo | is_ecg),
        (f"caption >= {cfg['min_caption_chars']} chars", caption.str.len() >= cfg["min_caption_chars"]),
    ]
    if cfg["exclude_later_timepoints"]:
        steps.append(("not a later-stage image (post-op, follow-up, healed)", ~caption.str.contains(LATER_TIMEPOINT)))
    if cfg["exclude_echo"]:
        steps.append(("not an echocardiogram", ~caption.str.contains(ECHO)))
    steps += [
        ("linked to a case", df.case_id.notna()),
        (f"age >= {cfg['min_age']}", df.age >= cfg["min_age"]),
    ]
    if cfg["require_ed"]:
        steps.append(("presented to emergency department", df.ed_presentation.fillna(False).astype(bool)))
    if cfg["min_vitals"]:
        steps.append((f">= {cfg['min_vitals']} measured vital sign(s) in text", df.n_vitals >= cfg["min_vitals"]))

    keep = pd.Series(True, index=df.index)
    funnel = []
    for name, mask in steps:
        keep &= mask.fillna(False).astype(bool)
        sel = df[keep]
        funnel.append({"step": name, "images": len(sel), "cases": sel.case_id.nunique(),
                       "articles": sel.article_id.nunique()})
    out = df[keep].copy()
    out["modality"] = np.select([is_rad[keep], is_photo[keep]],
                                [out.image_subtype, "acute_photo"], "ecg")
    out["is_echo"] = out.caption.fillna("").str.contains(ECHO)
    return out.reset_index(drop=True), pd.DataFrame(funnel)


def load_image_bytes(file_ids):
    """PIL images for a handful of file_ids (for display only)."""
    import io
    from PIL import Image
    out = {}
    for f in _image_files():
        t = pq.read_table(f, columns=["file_id", "image"], filters=[("file_id", "in", list(file_ids))])
        for fid, im in zip(t.column("file_id").to_pylist(), t.column("image").to_pylist()):
            out[fid] = Image.open(io.BytesIO(im["bytes"]))
    return out
