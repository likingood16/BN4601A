# Draft: vitals model section for the group report

*Draft for the group report (arXiv format, Arial 12 pt). About one page of text plus one table and two figures.
Numbers come from notebooks 01–03 (run 18 Oct 2026, split seed 42); re-check them if anything is re-run.
Items in [brackets] need the group's input.*

---

## Dataset & data preparation: vital signs

The vitals model was developed on the Emergency Service Triage dataset [1], which contains 1,267 adult visits to two
Korean emergency departments (Oct 2016 – Sep 2017). Each record holds the vital signs measured at triage (systolic and
diastolic blood pressure, heart rate, respiratory rate, temperature, SpO2), level of consciousness (AVPU), pain score,
age, sex and chief complaint. Its label is the Korean Triage and Acuity Scale (KTAS) level, from 1 (resuscitation) to
5 (non-urgent), assigned by a panel of three triage experts, together with the level given by the triage nurse.
KTAS 1–5 maps directly onto our priority scale, so this dataset provides the ground-truth triage labels that our image
and case datasets lack. It has no images or case identifiers and was used only to build and evaluate the vitals model.

Auditing the raw file showed placeholder strings for missing values (`??`, and `#BOŞ!` for missing pain scores),
decimal commas, and five visits recorded twice that differed only in a second diagnosis. These were merged, leaving
1,262 visits. One physiologically impossible value (SpO2 = 20% in an alert patient) was set to missing. A missing pain
score was set to 0 when the patient reported no pain. All 26 automated validation checks passed after cleaning
(valid codes, plausible ranges, diastolic < systolic, consistency of the recorded mistriage labels).

Missing data was not random. SpO2 was missing for 55% of visits overall. 48–64% of KTAS 1 patients had no
blood pressure, heart rate or respiratory rate recorded, against under 1% of KTAS 3–4 patients, because most were in
cardiac arrest or unresponsive. The visits were split once into training, validation and test sets (70/15/15,
stratified by expert KTAS; 882/190/190 visits). Columns known only after triage (diagnosis, disposition, length of
stay, nurse KTAS) were never used as inputs.

## Methods: vitals model

**NEWS2 with a missing-data safety extension.** We implemented the National Early Warning Score 2 [2] (SpO2 scale 1;
no supplemental-oxygen information was available, so room air was assumed). Standard NEWS2 counts an unmeasured
parameter as normal, which would rate a patient in cardiac arrest as low risk. We therefore added a rule that raises
the risk band to *high* when at least two of respiratory rate, systolic blood pressure and heart rate are missing, or
when the patient is unresponsive. The "unresponsive" condition follows KTAS itself, which places such patients at
level 1. The missing-vitals condition was suggested by the data audit.

**Classifier.** A gradient-boosting classifier [3] estimates `p_urgent`, the probability that the experts would triage
the patient KTAS 1–2. Inputs are limited to what our app receives: the six vitals, AVPU, pain score, age, sex, injury
and the NEWS2 total. The chief complaint was excluded because interpreting the history is the main language model's
job. Logistic regression was the comparison model. Because the validation set holds only 32 urgent patients,
hyperparameters and the alert threshold were chosen by 5-fold cross-validation on the training and validation sets
combined. The threshold was set to flag 80% of urgent patients (out-of-fold). The test set was used once.

**Safety floor.** The vitals model outputs a floor, the least urgent priority the main model may assign: priority 2 for a
high NEWS2 band, 3 for a medium band or any single parameter scoring 3, and 3 when `p_urgent` exceeds the threshold.
The floor is enforced in code after the language model answers (`final = min(LLM priority, floor)`). It can only
increase urgency, so it cannot introduce under-triage. A one-line text summary of all outputs is passed to the main
model.

## Results: vitals model

**Table X.** Detection of urgent patients (expert KTAS 1–2) on the held-out test set (n = 190, 37 urgent).
95% confidence intervals from 2,000 bootstrap resamples.

| Method | AUROC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|
| NEWS2, official trigger (≥ 5) | 0.69 (0.58–0.79) | 0.27 | 0.97 |
| Logistic regression | 0.79 (0.70–0.87) | – | – |
| Gradient boosting, p ≥ 0.14 | 0.78 (0.68–0.86) | 0.81 | 0.59 |
| Triage nurse (uses chief complaint) | 0.92 (0.84–0.97) | 0.84 | 0.97 |

- **Classifier vs NEWS2.** The classifier improved on NEWS2 (AUROC difference +0.085, 95% CI −0.002 to +0.169,
  paired bootstrap), consistent with cross-validation on the development set (0.83 vs 0.69). At its threshold it
  detected three times as many urgent patients as the official NEWS2 trigger, at the cost of more false alarms.
- **Calibration.** Predicted probabilities were well calibrated (Brier score 0.116 vs 0.157 for the base rate).
  NEWS2 total and SpO2 were the most important inputs (Figure Y).
- **Safety extension.** It raised the share of KTAS 1 patients rated high risk from 10 of 25 to 23 of 25. The cost
  was 21 escalations, 4 of them in KTAS 4–5 patients whose vitals had simply not been taken.
- **Floor coverage.** Combining the NEWS2 and classifier floors protected 30 of 37 urgent test patients, against 15
  with NEWS2 alone, while over-triaged visits rose from 3 to 28 of 190. [State the group's choice and why.]
- **Five-level prediction.** Predicting the full KTAS level from vitals alone was weak (quadratic weighted kappa
  0.41 vs 0.85 for the nurse), so the vitals model reports `p_urgent` and a floor rather than a triage level.

**External validation.** We applied the unchanged model to 8,412 adult visits from the US National Hospital Ambulatory
Medical Care Survey 2022 ED file [4], where urgency is the triage nurse's level (immediate/emergent = urgent; 18.6%).
The model did not outperform NEWS2 (AUROC 0.62, 95% CI 0.61–0.64, vs 0.62, 0.61–0.64), whereas the same settings trained on
US data reached 0.70. Hiding the SpO2-missingness signal, which differs sharply between the two datasets, changed AUROC by
less than 0.01 in either country, so it does not explain the gap. NEWS2 itself fell from 0.69 to 0.62, indicating that
vital signs are less tied to US nurse triage levels than to the Korean expert labels. Probabilities kept the right average
(17.7% predicted vs 18.6% observed) but were too spread out. We conclude that the NEWS2 formula and safety rule transfer,
while the learnt probability is dataset-specific and would require local recalibration (`figures/roc_external_nhamcs.png`).

The gap between vitals alone (AUROC ≈ 0.8) and the nurse, who also hears the chief complaint (≈ 0.9), shows that vital
signs cannot triage on their own. Many emergencies, such as chest pain or stroke, present with normal vitals. This
supports our design, in which the main language model combines the vitals note with the patient's history and image
findings.

*Figure suggestions:* `figures/roc_test.png` (ROC curves, test set); `figures/news2_risk_by_ktas.png` (NEWS2 risk band
by expert KTAS level); `figures/feature_importance.png` (appendix).

## Limitations: vitals model

- **Population.** Data from two Korean EDs (2016–17), adults only; KTAS thresholds and case mix may differ elsewhere.
- **Small sample.** Only 25 KTAS 1 visits overall and 37 urgent visits in the test set, hence wide confidence intervals.
- **Missing inputs.** SpO2 was missing for over half the visits, and no oxygen-therapy field exists, so NEWS2 may be
  underestimated by up to 3 points (flagged in the note).
- **Data-informed rule.** The missing-vitals rule was informed by the data audit.
- **Missing vs unmeasurable.** In our app a missing value may mean "not yet measured" rather than "unmeasurable",
  which the escalation rule cannot tell apart.
- **Labels.** The expert KTAS label is itself a judgement, not an outcome.

## Individual contribution (vitals)

[Name]: sourced and audited the triage dataset; built the cleaning and validation pipeline; implemented NEWS2 with
the missing-data safety extension; trained and evaluated the vitals classifier; designed the safety floor and its
interface to the main model; wrote the vitals sections of this report.

## References

[1] Moon S-H, Shim JL, Park K-S, Park C-S. Triage accuracy and causes of mistriage using the Korean Triage and Acuity
Scale. *PLoS One*. 2019;14(9):e0216972. Data via Kaggle: ilkeryildiz/emergency-service-triage-application.

[2] Royal College of Physicians. *National Early Warning Score (NEWS) 2: Standardising the assessment of
acute-illness severity in the NHS.* London: RCP; 2017.

[3] Pedregosa F, et al. Scikit-learn: Machine learning in Python. *J Mach Learn Res*. 2011;12:2825–2830.

[4] National Center for Health Statistics. *National Hospital Ambulatory Medical Care Survey: 2022 Emergency Department
Public-Use Data File and Documentation.* Hyattsville, MD: Centers for Disease Control and Prevention; 2024.
[Check the exact citation wording on the NHAMCS documentation page.]

[Add a KTAS reference if the report describes the scale in more detail.]
