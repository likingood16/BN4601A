# Vitals model (Model 2)

Turns a patient's vital signs into a NEWS2 score, a probability that the patient needs urgent care (KTAS 1–2), a
**safety floor** for the final priority, and a one-line note for the main language model.

Trained and evaluated on the Kaggle *Emergency Service – Triage Application* dataset (1,267 adult ED visits, Korea,
2016–17, expert KTAS labels). Original study: Moon et al., *PLoS One* 2019;14(9):e0216972.

## Using it in the app

```python
import sys; sys.path.insert(0, "vitals/src")
from vitals_model import VitalsModel, enforce_floor

vm = VitalsModel()                      # loads vitals/models/*
out = vm.assess(rr=26, spo2=89, sbp=112, hr=124, temp_c=38.6,
                consciousness="alert",  # alert / confused / verbal / pain / unresponsive
                age=71, sex="male", pain_score=2, injury=False, on_oxygen=None)

prompt_text = out["note"]               # put this in the main model's prompt
final = enforce_floor(llm_priority, out["floor_priority"])   # apply AFTER the main model answers
```

Every argument is optional; pass `None` for anything not measured. °F temperatures are converted, impossible values
(e.g. SpO2 9%) are treated as not recorded, and both are reported in `warnings`.

| Output key | Meaning |
|---|---|
| `note` | One line for the main model, e.g. *"NEWS2 9 = HIGH risk. Abnormal: RR 26/min (+3), SpO2 89% (+3)… Vitals model: 82% chance of needing KTAS 1-2 care… Safety floor: priority 2 or more urgent."* |
| `floor_priority` | Least urgent priority (1–5) the patient may get. 5 = no limit |
| `danger` | `True` when the floor is 1–2. The patient must not end up at the bottom of the list |
| `p_urgent` | Probability the experts would triage the patient KTAS 1–2 (calibrated on test data) |
| `urgent_flag` | `p_urgent` ≥ alert threshold (0.14, set to catch 80% of urgent patients) |
| `news2`, `news2_risk`, `subscores` | Standard NEWS2 score, risk band (with safety extension) and per-vital points |
| `escalated`, `escalation_reason` | Band raised to high because core vitals were unmeasurable or the patient is unresponsive |
| `missing`, `assumed`, `warnings` | What was not recorded, what was assumed (oxygen = room air), what was corrected |

**Safety rule:** the floor is applied in code after the main model, not in its prompt, so the language model cannot
override it. A floor can only make a patient more urgent.

**Group decision pending:** `VitalsModel(use_classifier_floor=False)` uses only the NEWS2-based floor. The default
(`True`) also floors any patient with `p_urgent` ≥ 0.14 at priority 3. On the test set that doubled the urgent patients
protected (15 → 30 of 37) but raised over-triage from 3 to 28 of 190 visits.

## Folder

```
data/raw/ktas_triage.csv           original Kaggle file (untouched)
data/processed/                    cleaned data, cleaning log, split, NEWS2 scores, test predictions
src/ktas_data.py                   load / clean / validate / split
src/news2.py                       NEWS2 + safety extension + note text
src/vitals_classifier.py           features and model definitions
src/vitals_model.py                what the app calls: VitalsModel.assess(), enforce_floor()
src/nhamcs.py                      maps the US NHAMCS 2022 ED file onto the same table (external validation)
data/external/nhamcs_2022/         NHAMCS raw file + scored output (not in repo)
models/                            trained models + model_meta.json (settings, threshold, test results)
notebooks/01…05                    cleaning → split + NEWS2 → classifier → app demo → US external validation
figures/                           report figures (300 dpi)
tests/test_vitals.py               python tests/test_vitals.py
```

**Get the data (not in the repo):**
1. Download the Kaggle dataset `ilkeryildiz/emergency-service-triage-application`, either from the website or with
   `kaggle datasets download ilkeryildiz/emergency-service-triage-application --unzip`.
2. Save its `data.csv` as `vitals/data/raw/ktas_triage.csv`.
3. Optionally, get the NHAMCS file for notebook 05 (see the Results section below).

To rebuild everything, run notebooks 01 → 05 in order (kernel `bn4601`), then `python tests/test_vitals.py`.
The saved models in `models/` already let the app run without the data.

## Results (test set: 190 visits, 37 urgent)

| Method | AUROC, KTAS 1–2 (95% CI) | Sensitivity / specificity |
|---|---|---|
| NEWS2 (official trigger ≥ 5) | 0.69 (0.58–0.79) | 0.27 / 0.97 |
| Vitals classifier (gradient boosting, p ≥ 0.14) | 0.78 (0.68–0.86) | 0.81 / 0.59 |
| Triage nurse (uses chief complaint) | 0.92 (0.84–0.97) | 0.84 / 0.97 |

Vitals alone miss many urgent patients (chest pain and stroke often present with normal vitals), so `p_urgent` and the
floor are inputs to the main model, not the triage decision.

**External validation (US, NHAMCS 2022 ED file, 8,412 adult visits, notebook 05):** the model does not beat NEWS2 outside
Korea (AUROC 0.62 vs 0.62; a model trained on US data reaches 0.70). The NEWS2 formula and safety rule transfer; the learnt
`p_urgent` is specific to the Korean data and would need recalibrating or retraining on local data. The NHAMCS file is
not in the repo: download `ed2022-stata.zip` from
`https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/NHAMCS/stata/` and unzip it into
`data/external/nhamcs_2022/raw/`.
