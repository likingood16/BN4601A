"""One rubric for all the group's models, each graded on its OWN held-out test results (nothing is re-run).

    python rubric.py            # prints the table (paste into REPORT_NOTES)

Grades: Poor / Acceptable / Good / Excellent. Each criterion has 3 cut-offs (Acceptable, Good, Excellent).
Where a published scale exists it is used and named; "own" marks cut-offs we chose, so the report must say so.
Base vs adapted are shown side by side where both were tested on the same examples (the brief asks for this).

Reads (all already saved):
  reasoning/outputs/metrics/  urgent__base_full__ktas_nhamcs.json, fusion__base4bit.json, level_model__base4bit.json,
                              answer_quality.json, answer_quality_stages.json, answer_quality_task.json, lora_eval_test.json
  vision/cato/hpc_run_3epochs/  evaluation_metrics.csv, test_predictions.csv, final_tests/openi/openi_tests.csv (Cato)
  vision/detector/runs/fracture/test_metrics.json                                              (fracture detector)
"""
import csv
import json

import pandas as pd

from config import OUTPUTS, REPO

M = OUTPUTS / "metrics"
CATO = REPO / "vision" / "cato" / "hpc_run_3epochs"
GRADES = ["Poor", "Acceptable", "Good", "Excellent"]

# published scales: cut-offs, higher is better, source
AUROC = (0.7, 0.8, 0.9), True, "Hosmer & Lemeshow 2013: 0.7 acceptable, 0.8 excellent, 0.9 outstanding"
KAPPA = (0.41, 0.61, 0.81), True, "Landis & Koch 1977: moderate, substantial, almost perfect"
UNDER = (0.25, 0.15, 0.05), False, "ACS-COT target <= 5% under-triage; 15% / 25% own"
READ = (60, 70, 80), True, "Flesch 1948: 60 plain English, 70 fairly easy, 80 easy (patient leaflets: 6th-8th grade)"
FRACTURE = (0.7, 0.8, 0.91), True, "Kuo 2022 meta-analysis: AI fracture detection pooled 91%; 70% / 80% own"
MEDQA = (0.5, 0.6, 0.7), True, "Singhal 2023 (Med-PaLM): 60% = pass mark"


def load(path):
    return json.loads(path.read_text())


def cato():
    with open(CATO / "evaluation_metrics.csv") as f:
        rows = {r[""]: r for r in csv.DictReader(f)}
    return {k: {"base": float(r["base (no adapter)"]), "adapted": float(r["adapted (best)"])} for k, r in rows.items()}


def cato_findings_only():
    """Cato's own files, re-scored on the FINDINGS text only, the one part the app uses (6 Nov 2026).
    His invented-details rate counts the impression + next-steps text too, which the app drops: all but 2 of his
    flagged cases are there. His Open-i 'normal_recognised' = 0 is a parsing problem: findings such as
    'Chest X-ray showing no acute process' were not counted as calling the X-ray normal."""
    pred = pd.read_csv(CATO / "test_predictions.csv")
    invented = {}
    for model, rows in pred.groupby("model"):
        hits = [any(fact[1].lower()[:40] in str(r["pred_findings"]).lower() for fact in json.loads(r["unsupported_facts"]))
                if str(r["unsupported_facts"]).startswith("[") else False for _, r in rows.iterrows()]
        invented["adapted" if model.startswith("adapted") else "base"] = sum(hits) / len(hits)
    openi = pd.read_csv(CATO / "final_tests" / "openi" / "openi_tests.csv")
    says_normal = openi["pred_findings"].fillna("").str.contains(
        r"no acute|unremarkable|normal|clear lungs|no evidence of", case=False)
    return invented, {"adapted": float(says_normal[openi["gt_normal"]].mean())}


def grade(value, cuts, higher=True):
    if value is None:
        return "-"
    passed = sum((value >= c) if higher else (value <= c) for c in cuts)
    return GRADES[passed]


def criteria():
    """(model, criterion, cut-offs, higher is better, source, {column: value})"""
    urgent = load(M / "urgent__base_full__ktas_nhamcs.json")
    fusion = load(M / "fusion__base4bit.json")
    level = load(M / "level_model__base4bit.json")
    base = load(M / "answer_quality.json")
    stages = load(M / "answer_quality_stages.json")
    task = load(M / "answer_quality_task.json")["task/stage1"]
    mixed = base["mixed/stage12"]
    qa = load(M / "lora_eval_test.json")
    vlm = cato()
    invented, normals = cato_findings_only()
    det = load(REPO / "vision" / "detector" / "runs" / "fracture" / "test_metrics.json")
    cut = load(REPO / "vision" / "detector" / "runs" / "fracture" / "threshold.json")["test"]   # cut-off set on validation

    def text(key, sub=None):
        """base / task adapter / mixed adapter on the same 490 patients"""
        get = (lambda d: d[sub][key]) if sub else (lambda d: d["all"][key] if "all" in d else d[key])
        return {"base": get(base["base"]), "task adapter": get(task), "mixed adapter": get(mixed)}

    m3s, m3l, m3t = "M3 urgency score", "M3 level + conformal", "M3 text (MedGemma)"
    rows = [
        (m3s, "AUROC, KTAS test (190)", *AUROC, {"result": urgent["test_ktas"]["auroc"]}),
        (m3s, "AUROC, NHAMCS test (1,262)", *AUROC, {"result": urgent["test_nhamcs"]["auroc"]}),
        (m3s, "under-triage of urgent, score + floor, KTAS", *UNDER, {"result": urgent["test_ktas"]["head_plus_floor"]["under_triage"]}),
        (m3s, "under-triage of urgent, score + floor, NHAMCS", *UNDER, {"result": urgent["test_nhamcs"]["head_plus_floor"]["under_triage"]}),
        (m3s, "calibration error (ECE), NHAMCS", (0.10, 0.05, 0.02), False, "own", {"result": fusion["test_nhamcs"]["text"]["ece"]}),
        ("M2 vitals (Caleb)", "AUROC, KTAS test, same patients", *AUROC, {"result": urgent["test_ktas"]["vitals_model_auroc"]}),
        ("M2 vitals (Caleb)", "AUROC, NHAMCS test, same patients", *AUROC, {"result": urgent["test_nhamcs"]["vitals_model_auroc"]}),
        (m3l, "weighted kappa vs nurse, KTAS", *KAPPA, {"result": level["test_ktas"]["safe_level"]["weighted_kappa"]}),
        (m3l, "weighted kappa vs nurse, NHAMCS", *KAPPA, {"result": level["test_nhamcs"]["safe_level"]["weighted_kappa"]}),
        (m3t, "under-triage of urgent, text model alone", *UNDER,
         {k: 1 - v for k, v in text("urgent_sensitivity").items()}),
        (m3t, "under-triage, text guided by score + floor", *UNDER,
         {"base": 1 - stages["base"]["guided"]["urgent_sensitivity"], "task adapter": 1 - task["guided"]["urgent_sensitivity"],
          "mixed adapter": 1 - stages["mixed/stage12"]["guided"]["urgent_sensitivity"]}),
        (m3t, "SBAR has all 4 headings", (0.9, 0.95, 1.0), True, "own; brief Core 3", text("all_4_headings", "sbar")),
        (m3t, "SBAR uses facts added in the chat", (0.8, 0.9, 0.95), True, "own; brief Core 3", text("report_uses_new_fact", "chat")),
        (m3t, "follow-up answer uses the new fact", (0.8, 0.9, 0.95), True, "own; brief Core 2 memory", text("followup_uses_new_fact", "chat")),
        (m3t, "patient-language reading ease", *READ, text("reading_ease_patient_turn3", "chat")),
        (m3t, "SBAR length, words", (500, 400, 300), False, "own: a handover read in about 2 minutes", text("mean_words", "sbar")),
        (m3t, "answers cut off while thinking", (0.05, 0.02, 0.0), False, "own", text("cut_off_while_thinking", "chat")),
        (m3t, "MedQA accuracy (200 test Qs)", *MEDQA,
         {"base": qa["base"]["medqa_acc"], "task adapter": qa["task/stage1"]["medqa_acc"], "mixed adapter": qa["mixed/stage12"]["medqa_acc"]}),
        ("M1 image (Cato)", "findings F1 vs case report", (0.2, 0.3, 0.4), True,
         "own; published X-ray report models score ~0.25-0.35", vlm["findings_f1"]),
        ("M1 image (Cato)", "invented details in the findings text (what the app uses)", (0.20, 0.10, 0.05), False, "own",
         invented),
        ("M1 image (Cato)", "box found for a named finding (IoU >= 0.5)", (0.3, 0.5, 0.7), True, "own", vlm["ground_ref_recall50"]),
        ("M1 image (Cato)", "normal chest X-rays called normal (Open-i, 20, re-scored)", (0.7, 0.8, 0.9), True, "own", normals),
        ("Fracture detector", "per-X-ray sensitivity (63 fractured)", *FRACTURE, {"cut-off 0.25": det["per_xray_sensitivity"], "cut-off 0.044 (app)": cut["sensitivity"]}),
        ("Fracture detector", "per-X-ray specificity (400 normal)", *FRACTURE, {"cut-off 0.25": det["per_xray_specificity"], "cut-off 0.044 (app)": cut["specificity"]}),
    ]
    return rows + image_rows()


# ---------------------------------------------------------------- the four ED image modalities (8 Nov 2026)
# Result files are copied from vanda into vision/results/ (fetch_image_results.sh); a missing file skips its rows.
RES = REPO / "vision" / "results"
CATCH = (0.8, 0.9, 0.95), True, "own: an ED safety net should catch >= 90-95% of time-critical findings"
BOXES = (0.3, 0.5, 0.7), True, "own (same scale as M1 boxes): box mAP50 / overlap >= 0.5"
SORT = (0.85, 0.9, 0.95), True, "own: a wrong sort means the right check never runs"
CHEST_CRITICAL = ["Pneumothorax", "Pleural effusion", "Consolidation", "Cardiomegaly"]


def caught(sens, spec):
    """A catch rate only counts if the cut-off still leaves most normal pictures alone: 'flag everything' catches 100%
    (8 Nov: the first rubric pass graded pneumothorax 'Excellent, 100% caught' at specificity 0)."""
    return (sens, "flags almost every picture") if spec is not None and spec < 0.5 else sens


def _get(name):
    p = RES / name
    return load(p) if p.exists() else None


def image_rows():
    rows = []
    chest = _get("chest_vindr.json")
    if chest:
        pi = chest["per_image"]
        rows += [("Chest X-ray (VinDr detector)", "box mAP50, all 14 findings (2,250 test)", *BOXES, {"result": chest["box_mAP50"]}),
                 ("Chest X-ray (VinDr detector)", "per-X-ray AUROC, ED-critical findings", *AUROC,
                  {k: pi[k]["auroc"] for k in CHEST_CRITICAL}),
                 ("Chest X-ray (VinDr detector)", "share caught at the validation cut-off", *CATCH,
                  {k: caught(pi[k]["sensitivity"], pi[k]["specificity"]) for k in CHEST_CRITICAL}),
                 ("Chest X-ray (VinDr detector)", "normal X-rays left unflagged (specificity)", (0.7, 0.8, 0.9), True, "own",
                  {k: pi[k]["specificity"] for k in CHEST_CRITICAL})]
    frac = _get("fracture_v2.json")
    if frac:
        pi = frac["per_image"]["fracture"]
        rows += [("Bone X-ray (fracture detector v2)", "box mAP50 (3,399 test)", *BOXES, {"result": frac["box_mAP50"]}),
                 ("Bone X-ray (fracture detector v2)", "per-X-ray AUROC", *AUROC, {"result": pi["auroc"]}),
                 ("Bone X-ray (fracture detector v2)", "per-X-ray sensitivity / specificity", *FRACTURE,
                  {"sensitivity": caught(pi["sensitivity"], pi["specificity"]), "specificity": pi["specificity"]})]
    ich = _get("ich.json")
    if ich:
        rows += [("Head CT (bleed classifier)", "per-slice AUROC, any bleed", *AUROC, {"result": ich["auroc"]["any"]}),
                 ("Head CT (bleed classifier)", "per-slice AUROC by bleed type", *AUROC,
                  {k: v for k, v in ich["auroc"].items() if k != "any"}),
                 ("Head CT (bleed classifier)", "bleed slices caught / normal slices unflagged", *CATCH,
                  {"caught": ich["any_sensitivity"], "unflagged": ich["any_specificity"]})]
    sort = _get("image_type.json")
    if sort:
        rows += [("Image sorter", "test accuracy, 7 types", *SORT, {"result": sort["test_accuracy"]}),
                 ("Image sorter", "accuracy by source", *SORT, sort.get("accuracy_by_source", {}))]
    ext = _get("external_tests.json") or {}
    if "fracture" in ext:
        rows.append(("Bone X-ray (fracture detector v2)", "per-X-ray AUROC by source (GRAZ children / FracAtlas adults)", *AUROC,
                     {k.split(" (")[0]: v["auroc"] for k, v in ext["fracture"].items() if isinstance(v, dict) and k != "all"
                      and v.get("auroc") is not None and not k.startswith("MultiCaRe")}))
    if "chest" in ext:
        cx = ext["chest"]["CheXpert classifier (no boxes), same tests"]
        rows += [("Chest X-ray: detector vs CheXpert classifier", "pneumonia set AUROC (other hospital, 624)", *AUROC,
                  {"detector": ext["chest"]["pneumonia set (children, 624)"]["auroc_highest_of_3"],
                   "classifier": cx["Pneumonia"]["pneumonia set"]["auroc"]}),
                 ("Chest X-ray: CheXpert classifier", "VinDr test AUROC, ED-critical", *AUROC,
                  {k: v["VinDr test"]["auroc"] for k, v in cx.items() if v["VinDr test"]["auroc"] is not None})]
    if "ich" in ext:
        pp = ext["ich"]["per_patient"]
        rows.append(("Head CT (bleed classifier)", "patients with a bleed caught / without left unflagged", *CATCH,
                     {"caught": pp["sensitivity"], "unflagged": pp["specificity"]}))
    if "imagetype" in ext:
        rows.append(("Image sorter", "other datasets sorted right (200 each)", *SORT,
                     {k.split(" ")[0]: v["correct"] for k, v in ext["imagetype"].items() if "correct" in v}))
    return [r for r in rows if r[-1] and all(v is not None for v in r[-1].values())]


def main():
    rows = criteria()
    print("| Model | Criterion (own held-out test) | Results -> grade | Cut-offs (Acc / Good / Exc) | Source |")
    print("|---|---|---|---|---|")
    for model, name, cuts, higher, source, values in rows:
        def one(k, v):
            name = "" if k == "result" else f"{k} "
            if isinstance(v, tuple):                    # (value, why it can't pass), see caught()
                return f"{name}{v[0]:.2f} **Poor** ({v[1]})"
            return f"{name}{v:.2f} **{grade(v, cuts, higher)}**"
        shown = "; ".join(one(k, v) for k, v in values.items())
        sign = ">=" if higher else "<="
        print(f"| {model} | {name} | {shown} | {sign} {' / '.join(f'{c:g}' for c in cuts)} | {source} |")


if __name__ == "__main__":
    main()
