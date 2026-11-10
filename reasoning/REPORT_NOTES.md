# Model 3 (MedGemma reasoning): notes for the group report

Everything the report needs about Model 3, in one place. Numbers come from `outputs/metrics/*.json`.
Report format (brief): arXiv style, PDF, 10 pages, Arial 12, with an individual-contributions section.
Due 16 Nov 2026, 23:59.

## 1. What Model 3 does (Introduction / Methods)
- **Specialty:** emergency medicine, focused on common emergencies (lecturer's guidance: trauma, cardiac
  arrest, stroke).
- **Model:** MedGemma 1.5 4B instruction-tuned (`google/medgemma-1.5-4b-it`). In training and on the HPC we
  used the identical, ungated copy `unsloth/medgemma-1.5-4b-it`.
- **Job 1, urgency score:** patient card → MedGemma vector (last layer, mean over all tokens) → logistic
  regression head → p(urgent) → Caleb's safety floor (can only raise urgency).
- **Job 2, writing (Core Tasks 2 + 3):** LoRA adapter for triage reasoning, chat and SBAR handover.

## 2. Data (Dataset & Data Preparation)
- **Pool 1 (urgency score):** KTAS (Korea, Kaggle) 1,262 patients, using Caleb's split (seed 42):
  train 882 / val 190 / test 190 (37 urgent). NHAMCS 2022 (US CDC, public domain) 8,412 adults:
  70/15/15 split stratified on triage level, seed 42.
  - Urgent = triage level 1–2: 19.4% of KTAS, 18.6% of NHAMCS.
- **Pool 2 (LoRA):** 386,425 chat examples, split train 347,779 / val 19,322 / test 19,324:
  - OpenMed SynthVision, with MultiCaRe-based records removed;
  - OpenMed MedDialog;
  - OpenMed Medical-Reasoning-SFT;
  - MedQA-USMLE (train split).

  Each example is tagged emergency or general.
- **Held-out tests:** 200 MedQA test questions; 5,690 MultiCaRe adult ED case reports (Caleb's ED filter).
- **Leakage guards:**
  - KTAS test = Caleb's exact 190 patients.
  - MultiCaRe-based SynthVision records removed.
  - MedQA benchmark questions removed from all training text.
  - Exact duplicates removed (MD5).
  - No diagnosis, disposition or nurse-triage columns used as inputs.
  - Caleb's vitals model refit on KTAS train+val only (his saved model had seen the test set).
- **Data check found 1 bad example** (two user turns in a row) out of 347,779. It crashed training at
  stage 6 and was fixed by merging the turns. Lesson: validate all data before long runs.

## 3. Methods
- **Patient card:** a fixed text layout (age/sex/arrival, complaint, history, observations, Caleb's
  NEWS2 + vitals-model note, imaging, notes). Missing fields are written as "not recorded".
- **Urgency head:**
  - Inputs and model: standardised MedGemma vectors → L2 logistic regression.
  - C chosen on KTAS validation by AUROC.
  - Alarm threshold set on validation to catch 80% of urgent patients (Caleb's rule); 90% also reported.
  - Test used once.
- **LoRA (QLoRA):**
  - Setup: Unsloth on the 4-bit base, r = 16, alpha = 16, dropout 0, attention + MLP of the language
    layers only (vision tower frozen).
  - Loss on assistant tokens only; max length 1,024 tokens; AdamW 8-bit; constant LR with 20 warm-up
    steps; 16 examples per update.
  - Staged training: chunks of 30,000 shuffled examples (seed 42). After each stage: validation loss
    (600 examples) and MedQA, saved as a learning curve.
  - Learning rate: 2e-4 for the first 22,448 examples (Google Colab T4, fp32); 1e-4 after that, because
    validation loss plateaued (2.179 → 2.182).
  - Compute: NUS HPC, NVIDIA A40 48 GB, bf16, about 50 min per stage. Colab's free T4 was 5× slower.
- **Three LoRA arms (what kind of data helps?):**
  - mixed = all 347,779 examples (12 stages);
  - emergency = emergency-tagged only (73,583);
  - two-stage = 30k general, then 30k emergency.
- **MedQA scoring:** compare the model's probability for A–D. Two formats:
  - "direct": the letter as the first answer word;
  - "suffix": after "The best option is".

  Each model is scored in its better format. Base does best with suffix (it explains first); the
  adapters do best with direct (trained to answer with the letter).

## 4. Evaluation (Experiments & Evaluation)
- **Urgency score:**
  - AUROC with bootstrap 95% CI;
  - Brier score and calibration error (ECE);
  - sensitivity, specificity, PPV, NPV, under- and over-triage at the 80% and 90% targets;
  - Spearman correlation with triage level;
  - decision curve analysis (Vickers & Elkin 2006);
  - subgroups (age 65+, sex);
  - 20 random re-splits;
  - learning curve.
- **LoRA, base vs adapted** (as the brief requires):
  - validation/test loss on assistant tokens, per source and emergency vs general, with paired bootstrap
    95% CIs;
  - MedQA accuracy with a paired CI.
- **Answer quality (`answers.py`, test data):**
  - Test A: triage decisions as JSON (190 KTAS + 300 NHAMCS test patients).
  - Test B: SBAR reports for 20 MultiCaRe cases.
  - Test C: a 4-turn consultation for 40 KTAS patients, matching Core Tasks 2 + 3:
    1. main concerns;
    2. a new fact (warfarin / penicillin allergy / kidney transplant / insulin);
    3. explain to the patient in simple words;
    4. SBAR "including everything we discussed".

    Checks: does the follow-up use the new fact? Does the report? SBAR headings, reading ease (Flesch),
    hedging, definitive-diagnosis phrases, words per turn.
  - MedGemma 1.5 "thinks" before answering (`<unused94>thought … <unused95>`). We score only the visible
    answer and report how often it was cut off while thinking.
- Testing plan: each brief requirement mapped to one test (A triage, A2 guided triage, B SBAR, C 4-turn chat), base vs adapters, test data only.

## 5. Results so far
**Urgency score** (head trained on KTAS + NHAMCS; tested once on the seed-42 test sets):

| | KTAS test AUROC | NHAMCS test AUROC |
|---|---|---|
| MedGemma vector + head, full precision | 0.901 (0.83–0.96) | 0.809 (0.78–0.84) |
| same, laptop 4-bit | 0.907 (0.83–0.97) | 0.814 (0.78–0.84) |
| head trained on KTAS only (4-bit) | 0.846 (0.75–0.93) | 0.703 |
| Caleb's vitals model (dev refit) | 0.779 | 0.610 |
| triage nurse (KTAS) | 0.92 | – |

- **At the 80% target** (KTAS / NHAMCS): sensitivity 0.84 / 0.74, specificity 0.83 / 0.75, PPV 0.54,
  NPV 0.95. Missed urgent: 6 of 37 / 61 of 235. Flagged 30% / 34% of patients.
- **At the 90% target:** sensitivity 0.89 / 0.83, specificity 0.71 / 0.63. Missed 4 / 40. Flagged 41% / 45%.
- **Benchmark:** the ACS Committee on Trauma aims for under-triage below 5%. Neither target meets it, so
  state this as a limitation.
- **Calibration:** Brier 0.077 (KTAS) and 0.120 (NHAMCS); ECE 0.066 and 0.020.
- **Decision curve:** the head beats "flag everyone" at every threshold from 5% to 30%.
- **Safety floor:** never flagged a patient the head had missed.
- **Subgroups (NHAMCS):** AUROC 0.77 for age 65+ vs 0.82 under 65 (a weakness); men 0.82, women 0.80.
- **20 re-splits:** MedGemma AUROC KTAS 0.917 ± 0.028, NHAMCS 0.810 ± 0.010. Vitals-only: 0.800 ± 0.035
  and 0.704 ± 0.017. **MedGemma won in 20 of 20 splits on both datasets.**
- **Learning curve (KTAS / NHAMCS AUROC):**
  - 10% of training patients: 0.880 / 0.783
  - 50%: 0.906 / 0.802
  - 100%: 0.907 / 0.814

  More patients now give only small gains.
- **Experiment 1:** adding raw vitals and NEWS2 to the head gave no gain (NHAMCS 0.814 → 0.812). MedGemma
  already reads the vitals in the card. Vitals alone: 0.811 / 0.695.

**LoRA:**
- **Mixed learning curve** (validation loss by stage, 1 → 12): 1.245, 1.139, 1.103, 1.076, 1.063, 1.047,
  1.038, 1.031, 1.023, 1.018, 1.009, 1.006. The curve flattens after about stage 6.
- **Final comparison** (full precision, validation, 100 examples per source):

| | loss all | emergency | general | MedQA |
|---|---|---|---|---|
| base MedGemma | 2.350 | 2.354 | 2.349 | 56.0% |
| **mixed (348k)** | **1.170** | **1.165** | **1.171** | 53.5% |
| emergency only (74k) | 1.286 | 1.205 | 1.305 | 52.5% |
| two-stage (30k + 30k) | 1.266 | 1.240 | 1.272 | 56.0% |

- **Mixed is best everywhere, even on emergency cases:** more varied data beat focused data.
- **MedQA did not improve.** LoRA teaches answer style and format, not new facts.
- **Laptop test-split trial** (3 per source): mixed lowers loss by about 1.09 [95% CI −1.24 to −0.97].
  The full test-split run is `lora_eval_test.json` (job `test_all`).
- **First answer-quality trial** (2 patients, laptop): the adapter writes valid JSON and short answers. The
  base model often hadn't finished thinking within 256 tokens, so the budget was raised. In one case the
  adapter's SBAR mentioned warfarin under Situation but said "no medication history" under Background:
  report internal inconsistency as a risk.
**Final tests on test data** (job `test_all`, 28 Oct; full precision; files `outputs/metrics/answer_quality.json`,
`outputs/metrics/lora_eval_test.json`, every answer in `outputs/answers/answers.json`):

*Base vs adapted on the Pool 2 test split* (200 per source; loss change vs base, paired 95% CI):

| | all | emergency | general | MedQA (vs base, 95% CI) |
|---|---|---|---|---|
| base | 2.381 | 2.488 | 2.352 | 56.0% |
| **mixed** | **1.202** (−1.18 [−1.21, −1.15]) | **1.218** (−1.27 [−1.36, −1.19]) | **1.198** (−1.15) | 53.5% (−2.5 [−10, +5]) |
| emergency | 1.310 (−1.07) | 1.269 (−1.22) | 1.320 (−1.03) | 52.5% (−3.5 [−11, +3.5]) |
| two-stage | 1.297 (−1.08) | 1.305 (−1.18) | 1.295 (−1.06) | 56.0% (0 [−7, +7]) |

This confirms on untouched test data: every adapter is clearly better at the answer style, mixed is best, and no
adapter changes MedQA.

*Triage decision written by the model* (JSON, 490 test patients: 190 KTAS + 300 NHAMCS):

| | answered | valid JSON | exact level | within 1 level | urgent sensitivity | under-triage of urgent |
|---|---|---|---|---|---|---|
| base | 92% | 80% | 41% | 84% | 0.34 | 66% |
| mixed | 100% | 100% | 50% | 94% | **0.18** | **82%** |
| emergency | 100% | 100% | 49% | 96% | 0.14 | 86% |
| two-stage | 100% | 100% | 48% | 95% | 0.16 | 84% |

- **Key finding:** the adapters almost always answer "level 3". Mixed gave level 3 to 385 of 490 patients and never
  1 or 5 (true counts: 12 level 1, 87 level 2, 236 level 3, 139 level 4, 16 level 5). That looks good on "within 1
  level" but misses most urgent patients.
- Example: a true level-1 chest-pain patient (pain 7/10, normal vitals) was called level 3 by both base and mixed.
- **Design conclusion:** a level written by the language model is unsafe. Urgency must come from the trained head
  (sensitivity 0.84 at the 80% target) + Caleb's floor. MedGemma explains and reports, but does not decide
  urgency. This justifies the two-part design.

*4-turn consultation* (Core Tasks 2 + 3; 40 KTAS test patients):

| | follow-up uses new fact | report uses new fact | SBAR headings | reading ease, clinical → patient | hedging | words per turn |
|---|---|---|---|---|---|---|
| base | 95% | 95% | 85% | 40 → 72 | 80% | 140 / 132 / 127 / 190 |
| mixed | 95% | 98% | 100% | 58 → 75 | 43% | 47 / 38 / 42 / 72 |
| emergency | 100% | 100% | 100% | 42 → 81 | 46% | 43 / 42 / 49 / 90 |
| two-stage | 100% | 93% | 100% | 46 → 81 | 45% | 50 / 44 / 47 / 86 |

- **All models remember and integrate the new fact** (93–100%), meeting Core 2's memory requirement and Core 3's
  "report uses the conversation" requirement.
- **All simplify for the patient:** reading ease rises to 72–81, plain English.
- **Adapters are 3× shorter and always use SBAR headings.** But they hedge about half as often (43–46% vs 80%), and
  reading the answers shows they are less thorough. Example (toe injury + type 1 diabetes): base flagged checking
  blood glucose and the missing SpO2; mixed only mentioned infection risk and foot care.
- Definitive-diagnosis phrases were rare in all models (≤ 0.6%). Base was cut off while thinking in 1.9% of chat
  turns; the adapters don't think aloud.
- **SBAR from 20 MultiCaRe case reports:** all four headings in 100% for every model; base about 490 words,
  adapters about 320–350.
- **Trade-off to discuss in the report:** LoRA learned the format (valid JSON, SBAR, brevity) but became less
  cautious and less thorough, and pulled triage levels towards the middle. This matches the brief's own example
  (the adapted model became concise, but not more accurate).

*With the urgency decision in the card (the app's design) + earlier mixed stages* (`answer_quality_stages.json`,
29 Oct; token limits raised to 2,048 so the base model can finish thinking; 490 test patients; chats with the
urgency line):

| | unaided: urgent caught | guided: never lowers flagged | guided: valid JSON | guided: all 6 fields | chat: follow-up / report use new fact | hedging | patient reading ease | words per turn |
|---|---|---|---|---|---|---|---|---|
| base MedGemma | 40% | 99.4% | **99%** | **62%** | **100% / 100%** | **89%** | 72 | 167 / 216 / 148 / 228 |
| mixed stage 1 | 25% | 100% | 9% | 21% | 100% / 95% | 35% | 81 | 43 / 40 / 41 / 100 |
| mixed stage 6 | 10% | 100% | 55% | 16% | 85% / 78% | 35% | 74 | 43 / 36 / 42 / 64 |
| mixed stage 12 | 18% | 100% | 99% | 35% | 98% / 88% | 40% | 73 | 47 / 35 / 41 / 71 |

- **Guided, every model catches exactly the urgent patients the score + floor catch (77%).** Models follow the
  decision and almost never raise a non-flagged patient, so safety comes from the ML part. The language model's
  job is to explain.
- **Earlier stages are not a better compromise.** Hedging drops already at stage 1 (89% → 35%), and chat memory
  worsens with more generic training (stage 6).
- **Base MedGemma is the best explainer so far:** 99% valid JSON, perfect chat memory, 89% hedging. It fills all six
  fields in only 62% of answers and is about 4× wordier.
- **Success rule fixed before the task-adapter results:** the app uses the task adapter only if never lowers
  flagged ≥ 99%, valid JSON ≥ 98%, all fields ≥ 80%, chat memory ≥ 95%, hedging ≥ 70%, and it is faster than base.
  Otherwise it uses base MedGemma.
- **Teacher data:** 1,857 of 2,000 triage examples kept (93%).

*ML level model (level_model.py; multinomial logistic regression on the same MedGemma vectors; C chosen by
5-fold CV on train; conformal sets at 90% from validation; tested once):*

| same 490 test patients | exact level | within 1 | weighted kappa | urgent sensitivity | urgent specificity |
|---|---|---|---|---|---|
| MedGemma base (writes level) | 41% | 84% | – | 0.34 | 0.80 |
| mixed adapter (writes level) | 50% | 94% | – | 0.18 | 0.96 |
| **ML level model** | **62%** | **96%** | 0.56 | 0.49 | 0.93 |
| **ML level, "safe" (≤ 2 when head + floor flag)** | 58% | 94% | 0.55 | **0.77** | 0.77 |

- **KTAS test:** exact 73%, weighted kappa 0.73. That is close to nurse-to-nurse agreement (0.77) and to nurse vs
  expert (0.83). NHAMCS: kappa 0.44.
- **Conformal sets:** the 90% guarantee holds on test (coverage 0.93 KTAS, 0.89 NHAMCS, 0.90 on the 490). But the
  sets are wide: about 78% of patients get 2 or more levels. That is honest but not very decisive. An 80% target
  would give smaller sets (trade-off to report).
- **Conclusion:** ML beats MedGemma at choosing the level. The level shown in the app should come from the ML
  model, made safe with the urgency decision.

## 5b. Discussion: why the LoRA didn't raise accuracy
1. **It improved what it was trained on:** imitating the answers (test loss 2.38 → 1.20). It was never trained
   to triage.
2. **Fine-tuning mainly teaches format and style; knowledge comes from pre-training.** See LIMA (Zhou et al.,
   NeurIPS 2023) and "LoRA Learns Less and Forgets Less" (Biderman et al., TMLR 2024). This fits MedQA staying
   at 53–56%.
3. **The training data contained no triage labels,** so triage accuracy could not improve.
4. **The data shaped its habits:**
   - MedDialog (62% of the data) is short, confident replies, so the adapter became brief and hedged less.
   - Imitation drifts to the most common answer, which explains "always level 3".
   - Training on answers beyond what the model already knows can increase confident errors (Gekhman et al.,
     EMNLP 2024).
5. **The brief's own LoRA example** (Qwen2.5-VL + VQA-RAD) found the same: accuracy unchanged, answers far
   more concise.
- **Fix:** task-matched data with true labels, balanced levels and teacher answers filtered for quality
  (`make_task_data.py`, the "task" arm).
- **Supporting triage studies (verified 29 Oct):**
  - Fine-tuning on a task-specific ED triage dataset (ESI, from health records) worked where generic training
    did not. In "Knowledge-embedded LLMs for emergency triage" (Knowledge-Based Systems, 2025) the fine-tuned
    models matched experienced emergency physicians and reached perfect recall on high-risk cases.
  - LLMs "collapse" the triage scale and rarely give the most urgent level: in EQUITRIAGE (arXiv 2605.03998,
    2026), one model gave ESI-1 in 1.6% of cases vs 16.8% in reality. This matches our "always level 3" finding.
  - LLM triage on ESI and KTAS data shows demographic bias, and the reference labels are themselves noisy (nurse
    assessments) (arXiv 2504.16273, 2025).
  - Contrast: on Manchester Triage System vignettes, LLMs leaned towards *over*-triage (JMIR 2024, PMID 38875696),
    whereas on our real ED records they *under*-triaged.
- **Caveat:** LIMA's claim (alignment mainly teaches format) is debated. Critics note it was judged by human
  preference, not accuracy. Our MedQA result agrees with it, but say "consistent with", not "proves".

## 5c. Extra image model: fracture detector + outlines (vision/detector, 30 Oct 2026)
- **What:** YOLO11s-seg (Ultralytics, COCO-pretrained) fine-tuned on FracAtlas (Abedeen et al., Scientific Data
  2023, CC BY 4.0). It draws a box and an outline around each fracture on a bone X-ray. It sits next to Cato's
  ED-VLM (which writes text) and doesn't replace it.
- **Data:** the official fractured split (574 / 82 / 63) plus normal X-rays added with seed 42 (574 / 82 / 400),
  so it also learns not to mark normal bones. 59 damaged normal JPEGs were skipped.
- **Training:** laptop RTX 3050, 60 epochs (~2 h, resumed once after an interruption), batch 8, 640 px. The best
  epoch (56) was chosen on validation.
- **Test results (63 fractured + 400 normal X-rays, used once):**
  | Measure | Value | Plain meaning |
  |---|---|---|
  | Box mAP50 / mask mAP50 | 0.38 / 0.33 | how well its boxes/outlines overlap the radiologist's |
  | Box precision / recall | 0.58 / 0.38 | 58% of its marks are real fractures; it finds 38% of the fractures |
  | Per-X-ray sensitivity (conf 0.25) | 0.54 (34/63) | flags about half of fractured X-rays |
  | Per-X-ray specificity | 0.96 | leaves 96% of normal X-rays unflagged (16 false alarms in 400) |
  | Sensitivity hand / leg | 0.61 (n=36) / 0.46 (n=26) | hip and shoulder had only 2 test images each: too few to judge |
- **Comparison:** the FracAtlas paper reports YOLOv8s-seg box mAP50 0.627 and mask 0.589, but on the
  **validation** set, without extra normal images and with no test result. Our best validation box mAP50 was 0.44.
  The numbers aren't directly comparable.
- **Reading:** this is a "second pair of eyes" that rarely raises false alarms but misses about half of
  fractures. A negative result must not rule out a fracture. Validation scores were still rising at epoch 60, so
  longer training could help.
- **Limitations:** small test set (63 fractured); no patient IDs in FracAtlas (the same patient may appear in two
  splits); bone X-rays only; not clinically validated.

## 5d. Image step: Cato's adapted model (design choice, 2 Nov 2026; not yet run in the pipeline)
- **Why:** the brief's workflow ends in deploying the adapted VLM. Cato's ED-VLM (MedGemma v1 + LoRA, language side
  only, 750 MultiCaRe images) is the group's image-trained adapter. The reasoning side stays on base MedGemma 1.5,
  because our tests showed it beat our text adapters.
- **Only its findings step is used.** On his test predictions (`vision/cato/hpc_run_750/test_predictions.csv`) the
  "impression + next steps" step copies case-report style and invents patient details: e.g. for a hip radiograph it
  wrote a 45-year-old man with alcohol abuse, and it reports treatments as already given. That must not reach a record.
- **Findings are short captions:** about 10 words vs 85 for base; findings F1 0.14 → 0.20, assessment
  term recall 0.25 → 0.20 (16 generated cases, his metrics). Some are empty of content ("Clinical picture of the patient.").
- **Planned test:** Base vs Adapted vs detector on the same 463 X-rays (`pipeline_check.py --part images`).
- **Costs:** two base models in the app (v1 and 1.5). Our prompt wording isn't his, because his training code isn't in
  the shared files.

## 5e. Speed: why the written answers are slow (laptop test, 2 Nov 2026)
- **What we saw:** on the laptop (RTX 3050, 4 GB, MedGemma 1.5 4B in 4-bit), one patient's full consultation (explanation,
  3 chat replies, SBAR) took up to about 30 minutes. A second patient had not finished after 35+ minutes, so the test
  was stopped. These are rough observations, not a timed benchmark, and no step was timed separately.
- **The fast part:** making the patient vector took about 0.9 s per patient on the same laptop, and the urgency and
  level models only add a small calculation. The decision is quick; the writing is what's slow.
- **Likely reasons (not measured one by one):**
  1. MedGemma 1.5 writes hidden thinking before each answer. It needs a high limit (2,048 tokens) or it gets cut off
     mid-thought and gives no answer at all, as we found in testing.
  2. Text is produced one token at a time, so long answers take time on a small GPU, and 4 GB leaves almost no spare
     memory (3.9 GB in use).
  3. One consultation means 5 long answers.
  4. Each chat turn re-reads the whole conversation, so later turns have more to process.
- **What this means:** the laptop speed is not usable live. Options: a larger GPU for the Space (the HPC A40 ran
  earlier jobs much faster, but this flow has not been timed there), streaming the words as they're written, showing
  the urgency result first, testing shorter or no thinking (quality must be re-checked), and a pre-run backup case.
- **Pipeline test status:** record check passed; score check matched saved results on 2 patients; the results check
  passed on 1 patient (result used in chat and SBAR, SBAR headings present). A crash on the second patient (unreadable
  explanation) was fixed, but the re-run was stopped before that patient finished, so the fix isn't confirmed.
  Cato's model has not yet read an image in the pipeline. The full HPC checks have not run (vanda under maintenance).

## 5f. App design: a triage desk, checked against real practice (`old/first_mockup/`, 2 Nov 2026)
- **First version** was four tabs built around the pipeline. Compared with published triage practice it was missing
  nurse override, a patient queue, time targets, a critical-look shortcut and repeat vitals, so it was rebuilt as one screen.
- **What the screen does, and why:**
  | Feature | Basis |
  |---|---|
  | Waiting list sorted by urgency, with waiting times | Paediatric ED tool needed waiting time, test status, age and a priority (PMC7501580) |
  | Critically ill button, no typing | CTAS starts with a "critical look" (CJEM 2004) |
  | Accept, or Change level with a required reason | Nurse judgment and override kept in e-triage (US patent 12148529); "human in control" (course Responsible AI slides) |
  | Complaint pick-list, with typing allowed | Drop-downs in e-triage; standard complaint lists (CEDIS, HL7 CTAS). Our list is our own short one |
  | Time to see a doctor and recheck interval by level | Canadian CTAS targets (PMC6975254, HL7 CTAS) |
  | Repeat vitals saved separately, first level unchanged | CTAS records reassessment separately (HL7 CTAS). Also why our arrival card is frozen |
- **Limits to state:**
  - Singapore uses PACS P1–P4 (KTPH); our models were trained on 5-level KTAS and NHAMCS labels. The PACS label on screen is
    our own conversion and is untested.
  - The time targets are Canadian; we found no public Singapore targets.
  - Evidence on AI triage screens is thin, and no nurse or clinician has tried this screen.
- **Testing:** 63 automated button checks passed in test mode (fake written answers and a fake urgency score), and most
  buttons were also clicked in a real browser. Browser testing found problems the automated checks missed: blank vitals
  sent as 0, a level choice carrying over between patients, and Accept able to lower a critically ill patient. All fixed.
  The real models have not yet run through the app.
- **Sources:** https://www.nhghealth.com.sg/find-care/Service/ktph-acute-emergency-care ;
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7501580/ ;
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12148529 ;
  https://terminology.hl7.org/en/ECS_MR-CTAS.html ; https://pmc.ncbi.nlm.nih.gov/articles/PMC6975254/table/t1

## 5g. Task adapter results + new image run (HPC jobs 1507022/1507023 finished 2 Nov; read 6 Nov 2026)
Task adapter = 1 epoch on 8,274 teacher examples (`task_train.jsonl`), val loss 0.772, 29 min on an A40.
Same 490 test patients (KTAS 190 + NHAMCS 300) as `answer_quality.json`.
| | base | task/stage1 | mixed/stage12 |
|---|---|---|---|
| exact level | 0.41 | 0.48 | 0.50 |
| within 1 level | 0.84 | 0.93 | 0.94 |
| urgent sensitivity, model alone | 0.34 | 0.37 | 0.18 |
| urgent sensitivity, guided by score + floor | 0.77 | 0.77 | 0.77 |
| SBAR words / all 4 headings | 486 / 1.00 | 318 / 0.95 | 328 / 1.00 |
| chat: follow-up uses new fact | 0.95 | 1.00 | 0.95 |
| chat hedging rate | 0.80 | 0.86 | 0.43 |
| chat words per turn | 127–190 | 134–185 | 38–72 |
| MedQA (200 test Qs) | 0.56 | 0.545 (diff −0.015, CI −0.085 to 0.055) | 0.535 |
- Reading: the task adapter levels like the mixed adapter but **keeps the base model's hedging and fuller answers**,
  and doesn't lose urgent patients the way mixed does (0.37 vs 0.18 alone). Guided by score + floor, all are equal
  (0.77), so the safety comes from the score and floor, not from the text model. Base was run twice (0.41 / 0.44 exact):
  differences of a few points are noise. No knowledge gain on MedQA.
- **Image step switched to Cato's `hpc_run_3epochs`** (5 Nov): MedGemma 1.5 base (same as Model 3), findings F1 0.25 vs
  0.20 for `hpc_run_750`, keeps the format; chosen over `hpc_run_5epochs` (F1 0.26) for fewer invented details on
  image-only questions (25% vs 34%). Not yet run inside our pipeline.

## 5h. Rubric: every model graded on its own held-out test (`training/rubric.py`, 6 Nov 2026)
Reads saved results only; re-run `python training/rubric.py` after new results. Grades Poor / Acceptable / Good / Excellent;
published scales are named, "own" cut-offs must be called ours in the report. Hosmer & Lemeshow call 0.8-0.9
"excellent" and >= 0.9 "outstanding"; our grade names sit one step lower.

| Model | Criterion (own held-out test) | Results -> grade | Cut-offs (Acc / Good / Exc) | Source |
|---|---|---|---|---|
| M3 urgency score | AUROC, KTAS test (190) | 0.90 **Excellent** | >= 0.7 / 0.8 / 0.9 | Hosmer & Lemeshow 2013: 0.7 acceptable, 0.8 excellent, 0.9 outstanding |
| M3 urgency score | AUROC, NHAMCS test (1,262) | 0.81 **Good** | >= 0.7 / 0.8 / 0.9 | Hosmer & Lemeshow 2013: 0.7 acceptable, 0.8 excellent, 0.9 outstanding |
| M3 urgency score | under-triage of urgent, score + floor, KTAS | 0.16 **Acceptable** | <= 0.25 / 0.15 / 0.05 | ACS-COT target <= 5% under-triage; 15% / 25% own |
| M3 urgency score | under-triage of urgent, score + floor, NHAMCS | 0.28 **Poor** | <= 0.25 / 0.15 / 0.05 | ACS-COT target <= 5% under-triage; 15% / 25% own |
| M3 urgency score | calibration error (ECE), NHAMCS | 0.02 **Excellent** | <= 0.1 / 0.05 / 0.02 | own |
| M2 vitals (Caleb) | AUROC, KTAS test, same patients | 0.78 **Acceptable** | >= 0.7 / 0.8 / 0.9 | Hosmer & Lemeshow 2013: 0.7 acceptable, 0.8 excellent, 0.9 outstanding |
| M2 vitals (Caleb) | AUROC, NHAMCS test, same patients | 0.61 **Poor** | >= 0.7 / 0.8 / 0.9 | Hosmer & Lemeshow 2013: 0.7 acceptable, 0.8 excellent, 0.9 outstanding |
| M3 level + conformal | weighted kappa vs nurse, KTAS | 0.66 **Good** | >= 0.41 / 0.61 / 0.81 | Landis & Koch 1977: moderate, substantial, almost perfect |
| M3 level + conformal | weighted kappa vs nurse, NHAMCS | 0.47 **Acceptable** | >= 0.41 / 0.61 / 0.81 | Landis & Koch 1977: moderate, substantial, almost perfect |
| M3 text (MedGemma) | under-triage of urgent, text model alone | base 0.66 **Poor**; task adapter 0.63 **Poor**; mixed adapter 0.82 **Poor** | <= 0.25 / 0.15 / 0.05 | ACS-COT target <= 5% under-triage; 15% / 25% own |
| M3 text (MedGemma) | under-triage, text guided by score + floor | base 0.23 **Acceptable**; task adapter 0.23 **Acceptable**; mixed adapter 0.23 **Acceptable** | <= 0.25 / 0.15 / 0.05 | ACS-COT target <= 5% under-triage; 15% / 25% own |
| M3 text (MedGemma) | SBAR has all 4 headings | base 1.00 **Excellent**; task adapter 0.95 **Good**; mixed adapter 1.00 **Excellent** | >= 0.9 / 0.95 / 1 | own; brief Core 3 |
| M3 text (MedGemma) | SBAR uses facts added in the chat | base 0.95 **Excellent**; task adapter 1.00 **Excellent**; mixed adapter 0.97 **Excellent** | >= 0.8 / 0.9 / 0.95 | own; brief Core 3 |
| M3 text (MedGemma) | follow-up answer uses the new fact | base 0.95 **Excellent**; task adapter 1.00 **Excellent**; mixed adapter 0.95 **Excellent** | >= 0.8 / 0.9 / 0.95 | own; brief Core 2 memory |
| M3 text (MedGemma) | patient-language reading ease | base 72.10 **Good**; task adapter 75.00 **Good**; mixed adapter 75.30 **Good** | >= 60 / 70 / 80 | Flesch 1948: 60 plain English, 70 fairly easy, 80 easy (patient leaflets: 6th-8th grade) |
| M3 text (MedGemma) | SBAR length, words | base 486.40 **Acceptable**; task adapter 317.80 **Good**; mixed adapter 327.90 **Good** | <= 500 / 400 / 300 | own: a handover read in about 2 minutes |
| M3 text (MedGemma) | answers cut off while thinking | base 0.02 **Good**; task adapter 0.00 **Excellent**; mixed adapter 0.00 **Excellent** | <= 0.05 / 0.02 / 0 | own |
| M3 text (MedGemma) | MedQA accuracy (200 test Qs) | base 0.56 **Acceptable**; task adapter 0.55 **Acceptable**; mixed adapter 0.54 **Acceptable** | >= 0.5 / 0.6 / 0.7 | Singhal 2023 (Med-PaLM): 60% = pass mark |
| M1 image (Cato) | findings F1 vs case report | base 0.12 **Poor**; adapted 0.25 **Acceptable** | >= 0.2 / 0.3 / 0.4 | own; published X-ray report models score ~0.25-0.35 |
| M1 image (Cato) | invented details in the findings text (what the app uses) | adapted 0.02 **Excellent**; base 0.02 **Excellent** | <= 0.2 / 0.1 / 0.05 | own |
| M1 image (Cato) | box found for a named finding (IoU >= 0.5) | base 0.00 **Poor**; adapted 0.12 **Poor** | >= 0.3 / 0.5 / 0.7 | own |
| M1 image (Cato) | normal chest X-rays called normal (Open-i, 20, re-scored) | adapted 0.80 **Good** | >= 0.7 / 0.8 / 0.9 | own |
| Fracture detector | per-X-ray sensitivity (63 fractured) | cut-off 0.25 0.54 **Poor**; cut-off 0.044 (app) 0.81 **Good** | >= 0.7 / 0.8 / 0.91 | Kuo 2022 meta-analysis: AI fracture detection pooled 91%; 70% / 80% own |
| Fracture detector | per-X-ray specificity (400 normal) | cut-off 0.25 0.96 **Excellent**; cut-off 0.044 (app) 0.73 **Acceptable** | >= 0.7 / 0.8 / 0.91 | Kuo 2022 meta-analysis: AI fracture detection pooled 91%; 70% / 80% own |

What it says, plainly:
- **Strong:** the urgency score (AUROC 0.90 KTAS / 0.81 NHAMCS, well calibrated), SBAR + chat memory, plain-language answers.
- **Weak:** under-triage is still 16-28%, far from the 5% target, and the text model alone misses most urgent patients
  (so the score + floor must lead and the text only explains). Caleb's vitals model alone is poor on NHAMCS (0.61).
- **Image side, re-scored on what the app uses (6 Nov):** Cato's adapter doubles findings F1 (0.12 -> 0.25). His
  36% "invented details" sits in the impression + next-steps text, which the app drops; in the findings text it is 2%,
  same as base. His Open-i "0 normals recognised" was a parsing problem: re-scored, 13-16 of 20 normal chest X-rays
  are called normal (65-80%, depending on whether "no evidence of pneumonia" counts). So no retraining was needed.
  Boxes (grounding) stay poor. The detector at the validation-chosen cut-off catches 81% of fractures.
- **Adapters vs base:** better format and length, same safety, no knowledge gain (MedQA 0.55 vs 0.56).
- References: Hosmer DW, Lemeshow S, Sturdivant RX. Applied Logistic Regression, 3rd ed. Wiley 2013. Landis JR, Koch GG.
  Biometrics 1977;33:159-74. American College of Surgeons Committee on Trauma, Resources for Optimal Care of the Injured
  Patient 2014 (under-triage < 5%). Flesch R. J Appl Psychol 1948;32:221-33. Kuo RYL et al. Radiology 2022;304:50-62.
  Singhal K et al. Nature 2023;620:172-80.

## 5i. Improvements from the rubric (6 Nov 2026; trial run on vanda queued: jobs 1525665 trial, 1525666 full)
- **Urgency cut-off 80% -> 90%** (`heads.py --add-threshold 0.90`, set on KTAS validation, 32 urgent patients):
  under-triage KTAS 16% -> 11%, NHAMCS 26% -> 17%; specificity 0.83 -> 0.71 / 0.75 -> 0.63; patients flagged
  30% -> 41% / 34% -> 45%. 95% would give ~9% under-triage but flag over half of all patients.
- **Fracture detector cut-off 0.25 -> 0.044**, set on 164 validation X-rays to catch 80% (`vision/detector/choose_threshold.py`):
  test sensitivity 0.54 -> 0.81, specificity 0.96 -> 0.73. It now points where to look; it never rules a fracture out.
- **Explainer = task adapter** (5g/5h), **one MedGemma in memory** with three roles (plain for the vector, task adapter for
  text, Cato's adapter for images), **no thinking block** (token `<unused94>` suppressed) and answer caps 768/512/768 tokens.
  Speed per step is measured by the trial (`pipeline_check_results.json`, "seconds").
- Not changed: the vitals floor, the level model, Cato's model (re-scored in 5h: its findings text is fine; no retraining).

## 5j. App redesign: chart first, AI second (6 Nov 2026; full design in APP_DESIGN.md)
- The client judged the Gradio "ED Case Board" too structured for a rushed ED. In the new design the nurse types one messy
  note (plus images) and presses Submit; the AI formats it.
- Why: in Hill et al. 2013, ED doctors spent about 43% of a shift on data entry. AI scribes save about 1–2 min per note,
  but note quality is under-studied, so our case file is editable and the AI text is labelled.
- Numbers are pulled out of the note by fixed rules (`src/charting.py`), not by MedGemma, because they feed the urgency
  model. They are shown as big NEWS2-coloured tiles with the text each came from, and the nurse can fix them and re-score.
- Images are read blind (no note given to the image model, to avoid anchoring). Detector boxes go automatically to MedSAM
  (detect → segment), and the note's body parts are written next to the reading.
- Stack: React screens (charting / case file / whiteboard) + FastAPI back end wrapping the same pipeline. One AI job at a
  time, and the whiteboard updates live on a second screen.
- Usability rubric (SUS, task success, time and clicks per patient, AI wait, whiteboard reading at 3 m, heuristic
  problems, faithful case file, useful questions, number extraction) is in APP_DESIGN.md section 7. Not yet measured.

- MedSAM outline test (vanda job 1525769, 69 FracAtlas fracture outlines, box = outline + 10% margin): median Dice
  **0.48** for MedSAM vs **0.39** for just shading the box. So MedSAM adds a little, but the overlap is moderate (one case
  0.07), and the outline is shown as "where to look", not a measurement. The chest check and sorter crashed (name clash
  between torchxrayvision and our src/model.py, then a Pillow blur error), were fixed 6 Nov, and are re-queued as job 1526190.
  The pipeline trial failed on Windows line endings in hpc.conf and is re-queued as 1526194, with the full check 1526195 after it.

- Image accuracy upgrades queued 6 Nov (no new data, same FracAtlas test so the numbers compare):
  - **Detector:** YOLO11m-seg at 1024 px on the A40, job 1526342. It was YOLO11s-seg at 640 px on the 4 GB laptop GPU.
    Fractures are tiny (about 0.2% of the picture), so resolution should matter most. Baseline to beat: box mAP50 0.379,
    and per-X-ray AUROC 0.856 (new cut-off-free number, added to choose_threshold.py).
  - **Outline:** MedSAM mask decoder fine-tuned on the FracAtlas train outlines with the encoder frozen and box jitter,
    epoch picked on val, job 1526343 (vision/detector/finetune_medsam.py). Baseline to beat: test Dice 0.48 (box only 0.39).
  - **Possible next step:** add GRAZPEDWRI-DX (Nagy et al. 2022, Sci Data; 20,327 paediatric wrist X-rays, 6,091
    patients, CC BY 4.0, 16.2 GB on figshare), split by patient, and still test on FracAtlas test. Waiting for the OK to download.

- Number extraction from messy notes (rubric row, APP_DESIGN.md section 10): 190 held-out KTAS test patients as
  template-written messy notes. Vital numbers 100% correct, 0 wrong or invented (rubric: Excellent, at least 99%).
  Template notes are easier than real ones, so this is an upper bound. Rule fixes (BIBA, "97.0F" read as age and sex,
  injury) were made on the train patients only, before the single test run.
- Claude's dry run of the 5 usability tasks: all completed; 7 problems found and fixed (APP_DESIGN.md section 10).
  SUS and human testers are still to do.

- **Vanda results (read 6 Nov, evening):**
  - **Chest check** on the Kermany children's pneumonia test set (624 X-rays, 390 pneumonia, not in training):
    - CheXpert weights: pneumonia AUROC **0.90**, sensitivity 0.81 / specificity 0.81 at 0.5;
    - "all" weights: AUROC 0.79, but sensitivity only 0.17 at 0.5;
    - both give lung opacity and consolidation AUROC 0.91–0.93.
    The app now uses the CheXpert weights.
  - **Full pipeline check** (job 1526195, 20 patients): all explanations valid; the decision is never changed by later
    results; chat and SBAR used the new results 20/20; no flagged patient was called non-urgent. Median seconds on an
    A40: decide 0.1, explain 10, image 0.9, detector 0.1, chat 9–15, SBAR 23; median SBAR length 238 words. The scores
    check (50 patients) gave identical cards and the same flag 50/50.
  - **Image upgrade jobs 1526342/1526343** failed because the FracAtlas train/val pictures weren't on vanda. They have
    now been uploaded and the jobs re-queued as 1527112 (detector) and 1527113 (MedSAM).
  - **Real-AI app job** 1527106 is queued; `connect_app.ps1` tunnels it to Chrome.

- **Image upgrade results (vanda, 7 Nov):**
  - **MedSAM fine-tuned on FracAtlas outlines** (mask decoder only, encoder frozen, box jitter, epoch 14 picked on val):
    test Dice **0.48 → 0.84** on the same 69 test fractures (box only: 0.39). Now the app's outline model
    (`image_tools.TUNED_MEDSAM`).
  - **Detector YOLO11m-seg at 1024 px:** worse than the small 640 px model. Box mAP50 0.33 vs 0.38, per-X-ray AUROC 0.854
    vs 0.856, sensitivity at 0.25 is 0.49 vs 0.54. More resolution and a bigger model didn't help with only 574 fractured
    training pictures, so the app keeps the original detector. (More data, e.g. GRAZPEDWRI-DX, is the next lever.)
- **App layout check (7 Nov):** an automatic overlap check ran at 375 / 1024 / 1280 / 1440 px on all three screens.
  - **Found:** the patient menu stuck out of its box (page overflow on phones); the sticky add bar covered the case file;
    the whiteboard table ran off the screen under 1100 px.
  - **Fixed:** all four widths now report 0 overlaps and 0 overflow. Every-button test still passes 56/56.

- **Real-AI check of the app, run 0** (vanda A40, 7 Nov; `outputs/real_ai_check_run0_before_fixes.md`):
  - **Text:** 10 held-out notes, 0 errors, median 34 s per patient. Both truly urgent patients were flagged. Level exact 4/8,
    within one level 7/8, with over-triage on 3 patients (one of them a possible-stroke flag).
  - **Images:** the AI got 9 of 10 picture types right (MRI was read as CT). The chest check gave sensible scores (pleural effusion
    pictures: lung opacity 77–81%); the bone X-rays were marked and outlined automatically.
  - **Problems found by reading every output:**
    1. Later notes and image readings never reached the case file (MedGemma's "file only the new notes" answer was unusable).
    2. Questions were empty for 6/10 (the JSON was cut off at 320 tokens) and generic or repeated otherwise.
    3. The raw note was copied into "Impression".
    4. The triage section was missing or rambling.
    5. Complaint lines included arrival mode or pain score.
    6. CT/MRI/other readings read like figure captions (Cato's adapter was trained on MultiCaRe captions): a known limit.
  - **Fixes:**
    1. Later notes are filed by fixed word rules, so nothing typed is lost or reworded.
    2. Question cap raised to 600 tokens, cut-off JSON still read, duplicates removed, questions specific to the complaint.
    3. Impression only when a clinician wrote one.
    4. Triage section filled from the trained models' decision, word for word.
    5. Complaint prompt asks for symptoms only.
  - Run 1 (after the fixes) is queued as job 1529034. The app job can now restart with new code without queueing again.

- **Real-AI check runs 1–3** (7 Nov, same 10 notes + 10 pictures; `outputs/real_ai_check_run{1,2,3}.md`):
  - **Run 1** (after the fixes above): image readings in the case file 10/10 (run 0: 0/10); patients with questions 20/20
    (run 0: 4/10); impression no longer a copy of the note; triage section from the decision.
  - **Run 2:** the patient line and the ED rules' pathway flags were given to MedGemma. The stroke patient was now asked
    "last known well time?" and "glucose?", and every flagged pathway got its to-do (ECG within 10 min, sepsis screen,
    mechanism of injury).
  - **Run 3:** questions and to-dos that only ask for values already recorded are dropped (e.g. "blood pressure?" with
    BP 193/115). What's left is genuinely missing (past history, medicines, allergies, SpO2).
  - **Stable across runs:** 0 errors, both truly urgent patients flagged, level within one 7/8, image type 9/10, a person's
    edit kept after a new note.
  - **Changed:** level exact moved 4/8 → 3/8 because the cleaner complaint line ("Dyspnea" instead of "Dyspnea walk-in")
    changes the card the urgency head reads. With 8 patients this is noise, but it shows the complaint wording feeds the score.
  - **Speed:** 12–44 s per patient on an A40, median 34 s.

## 5k. Which images to focus on: real ED data (7 Nov 2026)
- **NHAMCS 2022** (US national ED survey, 16,025 sampled visits = 155.4 million weighted visits; the same file as Pool 1):
  any imaging 53% of visits; **X-ray 37%**; CT 23% (**head 10%, abdomen/pelvis 10%**, chest 3%, other 6%);
  **ultrasound 6%**; MRI 1.5%. X-ray is used in 49% of injury visits; head CT in 19% of injury visits.
- **X-ray body parts** (not in NHAMCS): chest is the most common single radiograph, then lower and upper limb
  (urgent-care series 37% / 30% / 24%; paediatric ED series chest 28%, then ankle, foot, wrist, hand).
- **Most common acute head-CT finding:** haemorrhage, about a third of positives, then infarct and mass (adult ED series:
  15.5% of head CTs positive).
- **Focus** chosen from this:
  1. chest X-ray findings;
  2. limb X-ray fractures;
  3. head-CT bleed.

  CT abdomen and ultrasound are left out: there's no suitable labelled public data in the time available.
- **Models and data:**
  - **Image type:** EfficientNet-B0 on MultiCaRe pictures, split by article.
  - **Fractures:** YOLO11s on GRAZPEDWRI-DX (20,327 wrist X-rays, 6,091 patients, CC BY 4.0, split by patient) +
    FracAtlas; tested on the unchanged FracAtlas test set (old detector: box mAP50 0.379, per-X-ray AUROC 0.856) and the
    GRAZ test patients.
  - **Chest:** YOLO11s at 1024 px on VinDr-CXR (VinBigData Kaggle; 15,000 X-rays, 14 findings, 3 radiologists, boxes
    merged); per-finding AUROC and sensitivity at a validation-chosen cut-off.
  - **Head-CT bleed:** EfficientNet-B0 per slice on RSNA 2019 ICH (jhoward JPG version), split by patient; AUROC for
    any + 5 types. The location (Grad-CAM box + MedSAM) is not validated, since RSNA has no drawn boxes.
  - **Outlines:** each box goes to MedSAM for an outline (fine-tuned for fractures, plain otherwise).
- **Terms:** VinDr and RSNA are non-commercial, academic use; they stay on vanda and out of git.

### 5k-1. Chest X-ray detector result (job 1530462, finished 7 Nov 2026)
- **Training:** VinDr-CXR split 70/15/15 by X-ray (seed 42), test = 2,250 X-rays never seen in training. YOLO11s,
  1024 px, 80 epochs on an A40 (~2.6 min/epoch).
- **Boxes:** test box mAP50 = **0.365** (mAP50-95 0.182). For rough comparison, the top Kaggle VinBigData private leaderboard
  score was about 0.31 mAP40 (a different, hidden test set, so not a direct comparison; check the figure before citing). Boxes are good for large findings (aortic
  enlargement 0.94, cardiomegaly 0.93, pleural effusion 0.48) and weak for small or vague ones (other lesion 0.10,
  calcification 0.14, pneumothorax 0.15).
- **Per X-ray "is it there?"** (AUROC of the top box score):

  | Finding | Test X-rays with it | AUROC | Sensitivity / specificity at the val cut-off |
  |---|---|---|---|
  | Aortic enlargement | 474 | 0.99 | 0.84 / 0.98 |
  | Cardiomegaly | 367 | 0.98 | 0.82 / 0.98 |
  | Pleural effusion | 165 | 0.97 | 0.82 / 0.98 |
  | ILD | 56 | 0.97 | 0.71 / 0.98 |
  | Pleural thickening | 296 | 0.94 | 0.83 / 0.90 |
  | Pulmonary fibrosis | 235 | 0.93 | 0.76 / 0.95 |
  | Lung opacity | 201 | 0.91 | 0.79 / 0.93 |
  | Nodule/mass | 113 | 0.90 | 0.76 / 0.94 |
  | Infiltration | 82 | 0.89 | 0.61 / 0.97 |
  | Consolidation | 58 | 0.85 | see below |
  | Calcification | 62 | 0.84 | see below |
  | Pneumothorax | **15** | 0.83 | see below |
  | Atelectasis | 27 | 0.81 | 0.63 / 0.98 |
  | Other lesion | 188 | 0.80 | see below |

- **Problem found and fixed:** for 4 findings the detector draws no box at all on more than 20% of validation X-rays
  that have them, so "catch 80%" gave a cut-off of 0 (every X-ray flagged, specificity 0). The rule now uses the lowest
  real score instead and reports the lower sensitivity honestly (`reaches_80pct_on_val: false`). Re-scored without
  retraining (job 1530821); the first file is kept as `test_metrics_first.json`.
- **Clinical reading:** pneumothorax, the most time-critical chest finding, has only 81 training and 15 test X-rays in
  VinDr. Its result is not reliable, and the app must not present it as a rule-out.

### 5k-1b. Fracture detector v2 result + a scoring bug (job 1530469, finished 8 Nov 2026 00:53)
- **Boxes:** test box mAP50 = **0.927**, mAP50-95 0.552 (3,399 test X-rays: 2,936 GRAZ + 463 FracAtlas; early stop at
  epoch 58). The old detector scored 0.379 on FracAtlas alone, so the GRAZ wrists (children, one body part, many
  fractures) make up most of this. The per-source split is in the extra tests (`vision/eval_external.py`).
- **Bug in the per-X-ray score:** the first file said AUROC 0.45 with a cut-off of 0. The cause was the test code, not
  the model. It sent X-rays to the model in batches of 32 mixed-size images, and the answers came back out of step
  with the images. On the same 96 test X-rays, one at a time gives AUROC **0.975** and batches gave 0.34. The code now
  predicts one image at a time. Fracture and chest are re-scored without retraining (jobs 1531345, 1530821); the
  first files are kept as `test_metrics_first.json`.
- **Foreign body:** GRAZ labels a few foreign bodies, but none landed in the test set, so it isn't scored.

### 5k-1c. Image sorter v2: real hospital pictures added (8 Nov 2026)
MultiCaRe alone is journal figures (cropped, arrows, letters). The sorter decides which model runs, so a real X-ray
sorted wrong means no fracture or chest check at all. Added 700 train / 150 val / 150 test pictures from each of VinDr
(chest), GRAZ and FracAtlas (bone) and RSNA (head-CT slices). Each comes from that dataset's own split, so no test
patient is trained on. Test accuracy is reported per source (`labels_v2.csv`, `accuracy_by_source`).

### 5k-1d. Extra tests on other datasets (`vision/eval_external.py`, job 1531353 after the others)
- **Fracture:** GRAZ vs FracAtlas separately.
- **Chest:** detector vs the CheXpert classifier the app already has, on VinDr test, the pneumonia set (624
  children's X-rays, another hospital) and MultiCaRe captions.
- **Head CT:** per patient (any slice flagged).
- **Sorter:** 280 MultiCaRe + 200 each from VinDr, GRAZ, FracAtlas, RSNA and the pneumonia set.
- **Too small to mean anything:** the MultiCaRe caption tests (1 fracture caption out of 40 bone X-rays, 2
  pneumothorax, no head CT). Real external tests need more data (see the plan in the chat, 8 Nov).

### 5k-2. Stress test in Chrome (laptop, test mode, 7 Nov 2026 night)
- **Set-up:**
  - The app runs on the laptop with the fake AI (`server.py --test`).
  - `stress_test.py` sends 450 held-out patients (190 KTAS + 200 NHAMCS as messy notes, 60 MultiCaRe case
    openings), 8 at once.
  - At the same time, a person's actions are done by hand in Chrome: typing a stroke note, answering the AI's
    questions, adding a note to an existing patient through the picker, changing a stage, Clear, Put back.
- **Load:**
  - 0 errors out of 450. Every patient reached the board, most urgent first.
  - The board answers in about 0.1-0.2 s with 650 patients.
  - The browser stays responsive with 650 rows (under 10 ms lag).
  - With an open whiteboard refreshing every 3 s, the median time per patient goes from 8.7 s to 12.2 s (100 patients).
    That's acceptable.
- **Numbers:** 2,535 vital signs written into the notes, 100% pulled out correctly, none invented.
- **Board vs truth (urgency score + safety floor, no level model in test mode):** 81% of urgent patients flagged
  (KTAS 89%, NHAMCS 73%), specificity 51%.
- **Problems the hand checks found, all fixed and re-checked:**
  1. **"Already known" filter missed drug names.** The fixed stroke questions asked "on blood thinners?" with "on
     warfarin" in the note, and "any allergies?" with NKDA. The questions are now also checked against the note's own
     words (anticoagulant names, NKDA, glucose values, last known well).
  2. **GCS 14 shown as AVPU "verbal".** It is stored that way on purpose: NEWS2 scores new confusion like V, and
     Caleb's model only knows AVPU. But a clinician reads "verbal" as wrong, so the tile now shows ACVPU "C/V".
  3. **Case-file headings appeared twice after a second note.** This was a React key clash in the screen, not in the
     stored text.
  4. **Old screens after an update.** Browsers kept running the old screens because the page had no cache rule. It
     now sends `Cache-Control: no-cache`; one forced reload is needed the first time.
  5. **Short complaints were lost in test mode.** For example, "rt flank pain brought by family" was skipped as "just
     arrival", so the board showed "NRS on air alert". Junk complaints went from 19 to 9 of 450; the rest have no
     complaint written. The real AI writes the complaint itself.
  6. **BSL not filed under Investigations.**
- **Two usability problems, fixed on 8 Nov and checked in Chrome:**
  - The answer box moved down when the questions appeared, so a quick click could miss it. It now sits above them.
  - The patient picker was one long list (650 in this test). It now has a search box: bed, age/sex or complaint
    words, and Enter picks the top match. "chest pain" finds 6 of 80.
  - Button tests 56/56; 80-patient load 0 errors.
- **Tests after the fixes:**
  - `test_app.py` 56/56 pass.
  - Number checks: vitals still 100% on KTAS train (clean and messy).
- Results: `outputs/metrics/stress_chrome_load.json`, `stress_chrome_load2.json`, `stress_ab_noboard.json` and
  `stress_ab_board.json`.

### 5l. All-AI reasoning (git branch `ai-reasoning`, 8 Nov 2026; the rules version is kept on `rules-reasoning`)
The user asked for every reasoning step to be AI, even if slower. What changed:

| Step | Rules version (branch rules-reasoning) | AI version |
|---|---|---|
| ED pathway flags (stroke, chest pain, sepsis, ...) | words in the complaint (ed_assessment.PATHWAYS) | 7 trained heads on MedGemma's vector of the card (`pathway_heads.py`) |
| Pathway questions | fixed lists per pathway | MedGemma, steered by the AI flags |
| Numbers from the note | regex patterns | MedGemma reads them and quotes the words; a value whose words aren't in the note is left out with a warning |
| Filing later notes | keyword rules per heading | MedGemma picks the heading for each piece; the words go in as typed |

Unchanged: Caleb's vitals model and safety floor (Model 2, not ours to replace) and plain formatting steps.

**Pathway heads** (`pathway_heads.py`, `outputs/metrics/pathways__base4bit.json`):
- **Model:** logistic regression per pathway on the same base4bit vectors as the urgency head. C is chosen on
  validation, and the cut-off is set on validation to catch 90%.
- **Answer key:** the final diagnosis is in the pathway's group (NHAMCS ICD-10 DIAG1-5, KTAS diagnosis names), or the
  complaint names it. The diagnosis is never an input.
- **Test (1,452 patients), AI vs the old app's word rules:**

  | Pathway | Test patients | AUROC | AI caught / normal unflagged | Rules caught / normal unflagged | Hidden* caught, AI / rules |
  |---|---|---|---|---|---|
  | Chest pain | 147 | 0.91 | 0.84 / 0.84 | 0.74 / 1.00 | 39% / 0% |
  | Stroke | 21 | 0.93 | 0.90 / 0.81 | 0.67 / 0.96 | 87% / 53% |
  | Breathing | 190 | 0.96 | 0.90 / 0.88 | 0.71 / 1.00 | 66% / 0% |
  | Collapse/syncope | 46 | 0.92 | 0.89 / 0.82 | 0.39 / 1.00 | 82% / 0% |
  | Seizure | 24 | 0.95 | 0.92 / 0.89 | 0.75 / 1.00 | 67% / 0% |
  | Poisoning | 22 | 0.89 | 0.68 / 0.94 | 0.27 / 1.00 | 62% / 0% |
  | Infection | 189 | 0.87 | 0.92 / 0.56 | 0.42 / 1.00 | 86% / 0% |

  *Hidden = the diagnosis needed the pathway but the complaint words don't say so (e.g. "mental change" that was a
  stroke, "right hemiparesis" that the rule word "hemipleg" misses).
- **Reading it:** the AI catches more on every pathway, but raises more false flags (infection: 44% of patients
  without infection flagged). The rules' specificity of 1.00 is partly built in, since the answer key includes the
  complaint words. Poisoning (68%) is the weakest.
- **Small numbers:** stroke, seizure and poisoning have about 20 test patients each.

**Safety checks kept (checks, not reasoning):**
- a number whose quoted words aren't in the note is dropped;
- a filing answer that can't be read puts the piece under History, so nothing typed is lost.
- Tested with a fake model: an invented "RR 22" and a wrong "HR 120" are left out with a warning; a broken answer
  gives a warning, not a crash.
- Test mode (laptop, no MedGemma) still uses the rules as stand-ins. `test_app.py` 56/56.

**Full app, real AI, all-AI reasoning** (`outputs/real_ai_check_ai.md` on vanda, job 1531377; same 10 notes as run 4):
- **Same as the rules version:** 0 errors, urgent flagged 2/2, level within one 7/8, image type 9/10, imaging in the
  case file 10/10, edits kept.
- **Worse than the rules version:**
  - **Same patient written 3 ways:** no longer the same flag and level (urgency 0.12 / 0.14 / 0.46). AI number
    reading varies with the wording; the rules gave identical fields.
  - **False pathway to-dos:** "Chest pain: ECG" on anal pain, painful urination and a wrist injury; "possible
    stroke" on lightheadedness.
  - **Questions without the fixed pathway lists:** often generic or invented ("last known time of consciousness
    (LSTC)?", "patient's full name / date of birth?", "age?" when the note has it), as on 7 Nov.
  - **A missed BP in a long case report** ("Blood pressure on arrival was 113/67"), so the app asked for it.
- **Verdict:** not a success yet. The AI flags win on reworded notes, but in the full app they are too noisy, and the
  question writing and number reading are less reliable than the rules they replaced.

**Pathway flags, reworded notes (real AI, `pathways_reworded.json`, 15 test patients per pathway):**
- AI with the reworded complaint: chest pain 100%, stroke 100%, breathing 93%, collapse 100%, seizure 100%,
  poisoning 100%, infection 100%.
- Old word rules on the same reworded complaints: **0% on every pathway**.
- With the real complaints: AI 67-100%, rules 20-87%.
- **Problem:** 68% of test patients who needed no pathway got at least one flag (1.08 flags each on average).
  - **By pathway:** infection causes most of them (45%), then stroke 17%, collapse 16% and chest pain 13%.
  - **Fix 1, infection answer key:** narrowed to infections that can become sepsis (no more "any A/B code"). Infection
    false flags went 45% -> 36%, any false flag 68% -> 64%; infection caught 92% -> 87%.
  - **Fix 2, a bigger model:** a small neural net on the same vectors was no better (val AUROC 0.86-0.97 vs logistic
    0.88-0.98), so the limit is data, not the model. Stroke, seizure and poisoning have only about 90 training
    patients each.
  - **Next step:** more NHAMCS years (2019-2021, public domain, about 2-3 MB each), waiting for the user's OK.
  - **Cut-off trade-off on test:**

    | Setting | Any false flag | What it costs |
    |---|---|---|
    | Time-critical pathways 90%, others 80% | 55% | breathing caught 76%, infection 81% |
    | All pathways 75% | 30% | everything caught about 70% |

    Kept at 90% for now: a missed stroke costs more than an extra to-do.

**MedGemma number reading, first try** (190 KTAS test notes, `charting_numbers_test_ai_try1.json`):
- 88.9% of vital signs correct (rules: 100%); **0 wrong and 0 invented** vital signs, because the quote check works.
- Misses:

  | Problem | How often |
  |---|---|
  | Diastolic BP missing (it gave only the top number) | 48% |
  | Consciousness missing ("alert", "AVPU A", "GCS 15" not reported) | 33% |
  | BIBA not read as ambulance | 13 notes |
  | "Injured" claimed for illness pain (chest or abdominal pain) | 66 notes |

- **Second try** (job 1531532): clearer instructions (abbreviations, "alert counts", "injury = trauma only"), the BP
  quote split, and 600 tokens. Tuned on 150 TRAIN notes, then the test notes once. If it is still short of the
  rules, the next step is a small LoRA for number reading.

**Real-AI tests queued** (`hpc/jobs/ai_reasoning_check.pbs`, job 1531377):
1. MedGemma number reading on the 190 KTAS test notes, clean and messy (the rules score 100%).
2. Pathway flags on reworded complaints ("passed out at church", "fitting at home") vs the rules
   (`pathway_reworded_check.py`).
3. The full app, real-AI 10-note check.

**Number reading, second try results** (job 1531532, 190 KTAS test notes): clean 88.8% correct, 0 wrong or invented
(try 1: 88.9%); messy 89.4%, 1 wrong (try 1: 86.5%). The clearer instructions helped the messy notes a little, not the
clean ones; still short of the rules (100%).

**Pathway flags, structure test** (8 Nov, laptop CPU, `outputs/metrics/pathway_structures.json`): 7 layouts on the
same 1,452 test patients, every cut-off chosen on validation only. "Caught" = average over the 7 pathways; "hidden" =
patients who needed the pathway although the complaint words don't say so; "false" = share of the 944 patients who
needed no pathway that got at least one flag.

| Structure | Caught | Hidden caught | False | Flags per patient |
|---|---|---|---|---|
| Old word rules | 0.56 | 0.03 | 4% | 0.30 |
| 7 separate heads on the vector, 90% each (tries 1-2, in the app) | 0.86 | 0.73 | 64-69% | 1.58 |
| 7 heads on the complaint words only (TF-IDF) | 0.87 | 0.69 | 80% | 1.87 |
| 7 heads on vector + words | 0.86 | 0.71 | 61% | 1.41 |
| Same, 80% each | 0.75 | 0.45 | 33% | 0.73 |
| Time-critical pathways 90%, others 75% | 0.74 | 0.39 | 41% | 0.93 |
| Only the strongest 1 / 2 flags | 0.60 / 0.79 | 0.38 / 0.57 | 61% | 0.74 / 1.13 |
| One shared false-flag budget of 30% / 20% (on val) | 0.70 / 0.66 | 0.37 / 0.29 | 26% / 18% | 0.62 / 0.50 |
| **Gate "needs any pathway?" (90%) + 7 heads (90%), vector + words** | **0.81** | **0.60** | **31%** | 1.13 |

- **Chosen: gate + heads** (`pathway_heads.PathwayFlagger`, try 3, in the app now). It halves the false flags for a
  small loss in catch, and still catches more than the rules on every pathway. Per pathway, AI vs rules: chest pain
  0.87 / 0.74, stroke 0.86 / 0.67, breathing 0.89 / 0.71, collapse 0.80 / 0.39, seizure 0.88 / 0.75, poisoning
  0.64 / 0.27, infection 0.75 / 0.41.
- **Why not lower:** the false flags are often clinically arguable (dizziness or "mental change" flagged as possible
  stroke, cough as breathing, abdominal pain as infection), but the answer key counts them as wrong because the final
  diagnosis was something else. The limit is the labels and the data (88 stroke patients in training), not the model:
  a small neural network was not better.
- **Words are learned too:** the complaint-words part is a TF-IDF logistic regression trained on the same answer key,
  not hand-written rules.

**AI question writer** (8 Nov, `make_question_data.py`, `question_check.py`, job 1532070):
- **Problem:** in the full-app check the 4B model's own questions were often useless ("LSTC", "full name", "age?"
  with the age on the card).
- **Training data:** one example per Pool 1 patient (same train/val/test split).
  - Input: what the app gives the question step: the patient line, the AI pathway flags (the real ones for that
    patient, plus a wrong one 25% of the time), and a case file from a messy note. Some values, onset, allergies and
    medicines are written, some not.
  - Answer: the clinician questions for each pathway the patient really needed (final diagnosis), then the
    important missing values, then onset, allergies or medicines if not written. Never something already in the note.
  - So the model learns to ask about the real problem, and to ignore a flag the note doesn't support.
- **Adapter:** a LoRA started from the task adapter, used only for the questions step, 1 pass, 1,000 old examples
  mixed in.
- **Test:** 150 held-out test patients, current adapter vs trained one. Measures: valid answers; asks for something
  already known; junk (identity details or 1-word fragments); asked the real pathway's question. The targets
  themselves score 0.2% / 0% / 98% on this test, so the scorer is fair.
- **Result** (job 1532070, 22 min of training, val loss 0.325; `outputs/metrics/question_check.json`):

  | 150 held-out test patients | Current task adapter | Trained questions adapter |
  |---|---|---|
  | Valid answer | 100% | 100% |
  | Asks for something the note already has | 65% | **11%** |
  | Junk (name, address, 1-word fragment) | 0% | 0% |
  | Asks the real pathway's question (patients who needed one) | 20% | **78%** |

  MedQA (letter method) 51.5% vs 56% before; the adapter is used only for the questions step, so the explanation
  and SBAR are unchanged.

**Full app, real AI, all-AI try 2** (job 1532071, gate + heads flags, questions adapter; `outputs/real_ai_check_ai_try2.md`):
- 0 errors, urgent flagged 2/2, level exact 4/8 (rules run 4: 3/8), within one 7/8, image type 9/10, questions 20/20.
- **Faster:** median 26 s per patient (rules run 4: 38 s; try 1: 28-62 s).
- **Questions now clinical and specific** (e.g. chest pain: "spread to arm, jaw or back?", "known heart disease?";
  injury: "mechanism of injury?", "on blood thinners?"); no more "LSTC", names or "age?".
- **Pathway flags with the real model** (`pathways_reworded_try3.json`): patients needing no pathway with any flag
  **68% → 22%**; reworded complaints still caught 80-100% (rules 0%).
- **Still failing: the same patient written 3 ways.** Urgency 0.12 / 0.14 / 0.46 and the first is not flagged (rules
  version: 0.135 / 0.141 / 0.166, all flagged). The complaint words are the same as in the rules run, so the cause is the
  AI number reading (try 1: consciousness missed in 33%, diastolic BP in 48%); a missed value changes the card and
  can stop the vitals floor firing. Next: a small LoRA for number reading (training notes are made from known values).

**Fracture, adult cut-off** (job 1532072, cut-off set on 82 adult FracAtlas validation fractures, `external_tests.json`):

| 463 adult FracAtlas test X-rays (63 fractured) | AUROC | Caught | Normal unflagged |
|---|---|---|---|
| v2, shared cut-off (in 5k-3) | 0.843 | 22% | – |
| v2, adult cut-off | 0.843 | 89% | 50% |
| **v3 (adult X-rays repeated 4x), adult cut-off** | **0.879** | **82.5%** | **77%** |

v3 is better for adults and the same for children (GRAZ AUROC 0.989). The app knows the patient's age, so it can use
the adult cut-off for adults and the children's cut-off for children.

### 5k-3. Second rubric round, image models (8 Nov 2026)
- **Image sorter v2** (with hospital pictures):
  - 93.5% overall test accuracy: MultiCaRe 92.2%; VinDr, GRAZ and RSNA 100%; FracAtlas 99.3%.
  - Extra tests: 200/200 right on each of VinDr, GRAZ, FracAtlas and RSNA test pictures, and on the pneumonia set
    (never trained on).
  - MultiCaRe 280 held out: 91.1%. Bone X-ray is the lowest at 82.5%.
- **Fracture v2, re-scored:**
  - Per-X-ray AUROC 0.983, sensitivity 0.82 / specificity 0.99 at the validation cut-off.
  - **Split by source:** GRAZ children's wrists AUROC 0.986, caught 84%. FracAtlas adults AUROC 0.843, **caught only
    22%**.
  - The cut-off was set mostly by the 1,830 GRAZ fractures, so adults (most ED patients) are badly served.
  - **Fixes:** (1) a cut-off set on adult validation X-rays (`eval_external.py fracture_adult`); (2) fracture v3, with
    the adult training X-rays repeated 4x (`detector/prepare_v3.py`).
- **The rubric's own flaw (fixed):** the first pass graded "100% caught" as Excellent when it came from flagging every
  picture (specificity 0). Catch rates now only count when specificity is at least 0.5.
- **Chest, re-scored (per X-ray, one image at a time):**
  - The 4 findings that "flagged everything" now have real cut-offs: pneumothorax 0.53 / 0.99, consolidation
    0.71 / 0.98, calcification 0.69 / 0.95, other lesion 0.69 / 0.89 (caught / unflagged).
  - Other findings are unchanged: effusion 0.82 / 0.98, cardiomegaly 0.82 / 0.98.
- **Chest, other hospital:** the pneumonia set (624 children's X-rays, Guangzhou, never trained on). The detector's
  highest of consolidation / lung opacity / infiltration gives AUROC 0.89, caught 0.85, unflagged 0.82.
- **Chest, detector vs the CheXpert classifier on the same pictures:**
  - On VinDr test the classifier is worse: AUROC pneumothorax 0.73, effusion 0.86, cardiomegaly 0.87, pneumonia
    0.83, against the detector's 0.83 / 0.97 / 0.98 / 0.85.
  - On the pneumonia set they're about equal (0.90 vs 0.89).
  - So the detector decides and draws the boxes; the planned switch to the classifier was rejected.
- **Head CT, per patient** (any slice above the validation cut-off), 2,558 RSNA test patients:
  - AUROC 0.988; **97.1% of patients with a bleed caught**, 89.9% of patients without one left unflagged.
  - Caught by type: epidural 97.6%, intraparenchymal 98.2%, intraventricular 99.8%, subarachnoid 99.2%, subdural
    97.3%.
  - Caveat: this RSNA copy has about 50% bleed slices, more than a real ED.
- **MultiCaRe caption tests:** too few positives (1-3 per finding) to mean anything; reported only as such.
- **Grades now:**
  - **Head CT:** Excellent/Good.
  - **Sorter:** Good/Excellent.
  - **Fracture:** Excellent for children's wrists, Poor for adults → v3 training (job 1531401).
  - **Chest:** Good to Excellent for effusion, enlarged heart and pneumonia; **Poor for pneumothorax** (53% caught).
    Fixing that needs pneumothorax training data (the SIIM-derived set, 4.8 GB, waiting for the user's OK).

### 5m. Learning from use (`src/learning.py`, 8 Nov 2026, branch ai-reasoning)
- **What the app collects while people work** (`web/case_files/learning/`, gitignored with the board):
  - a nurse's number fix (the note, what the AI read, the correction);
  - an AI to-do ticked or removed (monitoring only);
  - when a patient is cleared: optional **final diagnosis** and **level the clinician judges was right**, typed in two
    small boxes next to "Clear from board", plus what the AI said at arrival and MedGemma's vector of the card.
- **What the update does** (`python learning.py --update`, CPU, minutes):
  - retrains the urgency head and the pathway flagger with the same methods as before, on the original training
    patients + 80% of the patients seen in use;
  - cut-offs are still chosen on the original validation patients;
  - the other 20% of in-use patients are kept to report how the candidate does on real use.
- **Safety gate, fixed in advance**, on the original frozen test patients. The candidate replaces the current model
  only if:
  - urgency AUROC falls by at most 0.01 on KTAS and on NHAMCS;
  - no fewer urgent patients are flagged;
  - pathways caught fall by at most 2 points;
  - false pathway flags rise by at most 2 points.
  Old files are kept (`models/heads/versions/`); `--rollback` restores them; every run goes to
  `outputs/learning/updates.json` and `CHANGELOG.md`; the whiteboard shows the model version.
- **Design choices (for the report):**
  1. No live learning inside the app. A model that changes mid-shift can't be checked. This follows the FDA's
     "predetermined change control plan" idea (final guidance Dec 2024): what may change, how it is tested, and
     what must not get worse are written down before any change.
  2. Labels come from the final diagnosis and level, never from "the nurse accepted the AI's to-do". Otherwise the
     AI teaches itself to repeat its own suggestions (a feedback loop, automation bias).
  3. Number fixes become training data for the number-reader adapter (GPU job), not for the heads.
- **Tests:**
  - App button test 61/61, including 5 new learning checks.
  - Plumbing check (40 fake in-use patients copied from training patients, models in a temporary folder): fewer
    than 20 patients → nothing changes.
  - With 40 patients the gate **refused** the candidate: 2 more urgent NHAMCS test patients missed (sensitivity
    0.830 → 0.821) although AUROC rose. This shows the gate working.
  - With the gate loosened (test only): switch worked, and the rollback restored the exact files.
  - This is a check of the mechanism, not a learning result. A real one needs new patients, e.g. NHAMCS 2019-2021
    used as "patients seen in use".

### 5n. Whole-app evaluation, all branches, vs deployed systems (8 Nov 2026)
**Branches** (laptop button test = every button through the back end, test mode; real-AI = vanda A40, 10 notes +
10 pictures):

| Branch | What it is | Button test | Real-AI full app |
|---|---|---|---|
| `reasoning` / `ed-app-v2` (2 Nov) | first one-screen Gradio desk, form input | not re-run (needs the old Gradio app running) | pipeline check 20/20 valid (5i) |
| `fixed-models` (7 Nov) | chart-first React + FastAPI app, rules reasoning, tuned MedSAM | 55/56* | runs 0-3: 0 errors, urgent 2/2, within one 7/8 |
| `rules-reasoning` (8 Nov) | + stress-test fixes; new image models trained but not wired in | 55/56* | run 4: 0 errors, 38 s/patient, same patient 3 ways: same flag and level |
| `ai-reasoning` (working tree, uncommitted) | AI pathway gate + heads, questions adapter, AI number reading and filing, learning from use | **61/61** | try 2: 0 errors, 26 s/patient, level exact 4/8, clinical questions; **same patient 3 ways: not consistent** |

*The 1 failure is "screens are served": the built screens aren't in git, so the scratch copy had none. All
functional checks pass. First runs failed the image step because the laptop ran out of memory (2.6 GB free) with
extra servers open; after closing them all passed.

**Other tests (8 Nov):** Caleb's vitals tests 8/8; number reading by rules 100% (clean and messy, 190 KTAS test
notes), by AI 88.8% / 89.4% with 0 / 1 wrong. Stress test, current tree (450 held-out patients, 8 at once, test
mode, `stress_laptop_8nov.json`): 0 errors, numbers 2,535/2,535, board 41 ms, urgent caught 81% (KTAS 89%, NHAMCS
73%; test mode scores with Caleb's vitals model, not the MedGemma head).

**Against published deployed systems** (same kind of task; their tests are bigger and often vendor-funded):

| Task | Ours (held-out test) | Deployed / published | Main difference |
|---|---|---|---|
| ED urgency | AUROC 0.90 KTAS, 0.81 NHAMCS; under-triage 11% / 17% at the app's cut-off | e-triage (Levin 2018, 172,726 visits): AUROC 0.73-0.92; Raita 2019 (NHAMCS 2007-15, 135,470 visits): critical care 0.86, hospitalisation 0.82; KATE: ESI accuracy 76% vs nurses 60% | similar discrimination, ~15x less data, our label is the nurse's level not the outcome; nobody meets the 5% under-triage target; no prospective test for us |
| Fracture X-ray | children's wrist AUROC 0.989; adults v3 0.879, caught 82.5% / 77% normal unflagged | BoneView: 90.9% / 82.1% or 79.2% / 96.5% (FDA); children 91.3% / 90.0%; Kuo 2022 meta-analysis pooled 92% / 91% | adults behind (574 adult fractures vs tens of thousands); children on par |
| Chest X-ray | effusion AUROC 0.97, cardiomegaly 0.98, pneumothorax 0.83 (caught 53%) | Annalise pneumothorax AUROC 0.979, 93.9% / 92.2% (FDA, 949 X-rays); Lunit ED AUROC 0.88-0.999 | pneumothorax far behind: VinDr has few pneumothorax cases; Annalise trained on ~800k X-rays |
| Head CT bleed | per patient AUROC 0.988, caught 97.1%, 89.9% unflagged | Aidoc FDA 93.6% / 92.3% and 96.2% / 94.8%; independent sites 84-99% sensitivity, PPV ~81% | on par on RSNA, but RSNA is ~50% bleeds; at real ED rates (a few %) most flags would be false, as for Aidoc |
| Notes → numbers | rules 100%, AI 89% on template notes | ambient scribes (DAX, Abridge): omissions the most common error, hallucinations rarer but present (NEJM AI 2025 RCT; Frontiers 2025) | ours reads typed notes, not speech; template notes are easier than real ones |
| LLM triage level | MedGemma alone misses most urgent (under-triage 63-82%), so it never decides | GPT-4 ESI accuracy 70-77% in small studies, tends to over-triage | we keep the decision in the trained head + floor |

**Is it a good clinical program?** A strong research prototype, not ready for clinical use:
- **Good:** the urgency score is in the published range; the bleed and children's fracture models are on par with
  cleared products on public test sets; the language model never decides urgency; every number shows its source;
  every AI item is marked; the record keeps who did what; models change only through a tested update.
- **Not yet:**
  - under-triage is 11-17%, against a 5% target;
  - the best image models aren't in the app yet;
  - pneumothorax and adult fractures are behind deployed tools;
  - AI number reading is 89%, and it makes the same patient score differently;
  - every test is retrospective, on public data and template notes;
  - no clinician has used it (no SUS or task study yet);
  - no regulatory or data-protection process.
- **Next step a hospital would want:** a silent (shadow) prospective run, where the app scores real patients but
  nobody sees the output, compared with what actually happened.

### 5o. Fix plan: what, why, and the research behind it (8 Nov 2026; deadline 16 Nov)
**Priority 1: no new data needed**

| # | Fix | Why (our result) | Research behind it | Done when |
|---|---|---|---|---|
| 1 | Put the new image models in the app: sorter v2, fracture v3, VinDr chest detector (keep CheXpert for pneumonia), head-CT bleed model, MedSAM outlines on the boxes | App still runs detector v1 (caught 81% / 73% normal unflagged) and has no head-CT model. The new ones score better on held-out tests (adult fracture AUROC 0.879, bleed 0.988, effusion 0.97) | Deployed tools use one model per task (Aidoc for bleeds, BoneView for fractures, 5n). MedSAM turns a box into an outline (Ma 2024). Heat-map boxes are rougher than a radiologist's (Saporta 2022), so the app labels them "rough area" | App test passes with all models; 10-picture real-AI check |
| 2 | Fracture cut-off chosen by age (child / adult) | One cut-off for everyone: v2 left only 50% of normal adult X-rays unflagged. Adult cut-off on v3: 82.5% caught, 76.7% unflagged | One cut-off for all groups can miss more in one group (Seyyed-Kalantari 2021). Most fracture-AI evidence is in adults; results in children differ and need their own check. TRIPOD+AI asks for results per group (Collins 2024) | Each cut-off chosen on validation, reported on test, per age group |
| 3 | Number reader: train a small "numbers" adapter on generated notes; compare it with the rule reader on every patient; if they disagree, show both and ask the nurse | AI reading is 89% vs rules 100%. The same patient written 3 ways gave urgency 0.12 / 0.14 / 0.46 (it missed consciousness or diastolic BP) | Fine-tuned small models match or beat prompted large ones for clinical extraction (ELMTEX 2025; QLoRA clinical extraction 2025). Wording changes alone move LLM accuracy (Sclar 2024), so a consistency test is needed. Leaving things out is the most common scribe error (5n) | ≥ 99% fields right, 0 wrong numbers, same patient 3 ways → same flag and level |
| 4 | Pneumothorax cut-off retuned on validation for higher catch | Only 53% caught | Cut-offs are chosen on validation and reported on test (TRIPOD+AI). Cleared tools run at about 94% catch (Annalise, 5n). Retuning only moves along the same curve (AUROC 0.83), so fix 6 is the real fix | Catch ≥ 80% on test, with the false-flag cost reported |
| 5 | Add a "95% catch" urgency cut-off and report what it costs | Under-triage is 11-17% at the app's cut-off | The ACS target is under-triage < 5%, with 25-35% over-triage accepted. Decision curves show whether the extra flags are worth it (Vickers 2006) | Under-triage and over-triage on KTAS + NHAMCS test |

**Priority 2: needs downloads (ask first)**

| # | Fix | Why | Research | Size |
|---|---|---|---|---|
| 6 | SIIM-ACR pneumothorax X-rays (on vanda) | VinDr has few pneumothorax cases; Annalise trained on about 800k X-rays | More examples of the rare finding is the main lever (5n comparison) | ~4.8 GB |
| 7 | NHAMCS 2019-2021 (CDC, public domain) | More urgent patients; a real "learning from use" test: older years = training, newer years = "seen in use" | Raita 2019 reached AUROC 0.86 with 135k NHAMCS visits. Patients and coding change over time (Finlayson 2021), so testing on later years (temporal validation, TRIPOD+AI) is fairer than a random split | ~2-3 MB each |
| 8 | Adult fracture X-rays (Bone Fracture Multi-Region), head-CT sets | 574 adult fractures vs tens of thousands for cleared tools | Same as 6 | 505 MB; 26 MB; 95 MB |

**Priority 3: prove it works**

| # | Step | Why | Research |
|---|---|---|---|
| 9 | Re-run the full app with the real AI on vanda + a real-AI stress test, then rubric round 3 | Laptop tests are test mode; most parts changed since the last real run | Test the whole system, not just each model (DECIDE-AI, Vasey 2022) |
| 10 | Usability test: group members run 5-10 cases each; time, errors, SUS score | No person has used the app yet | DECIDE-AI asks for human-factors results in early clinical studies. SUS (Brooke 1996); about 68 is average (Bangor 2009) |
| 11 | Report a silent trial as the next real-world step (not doable before 16 Nov) | All tests are retrospective | Silent trials catch drift and bias before patients are affected (Kwong 2022) |

**Why learning from use is built this way (already done, 5m):**
- Updates happen offline, behind a fixed gate: FDA PCCP guidance (Dec 2024).
- Labels come from outcomes, not from accepted AI suggestions. If clinicians act on a model's output, retraining on that data can make the model worse (Adam 2020; Vaid 2023). People also tend to accept AI suggestions without checking (automation bias, Goddard 2012).

### 5p. Fix plan, priority 1 done (8 Nov 2026 evening, branch ai-reasoning, uncommitted)
**1-2. New image models in the app** (`src/image_models.py`, CPU, used by the real pipeline and by test mode):
- **Sorter v2** replaces MedGemma's guess.
- **Fracture v3** with an age cut-off: child < 18 uses 0.6636; adult or age unknown uses 0.0137.
  - Check on the 63 FracAtlas adult test fractures: 52 marked (82.5%), as in 5o.
  - IMG0003297 is one of the misses (it was the old app test's picture).
- **VinDr chest detector**, one cut-off per finding (80% catch on validation).
  - Only ED findings are red and boxed: pneumothorax, effusion, pneumonia-type shadowing, cardiomegaly, aortic
    enlargement, nodule/mass, atelectasis.
  - Chronic findings are listed as "also seen". The first laptop test flagged 4-6 findings per abnormal X-ray.
  - Pneumonia uses the detector, not CheXpert: on the children's pneumonia set 85% / 82% vs CheXpert 97% / 51%.
- **Head-CT bleed model**, plus a Grad-CAM box ("rough area"), runs only when the notes mention the head (CT head,
  fall, GCS, headache ...), so a body CT never gets a bleed score.
- **MedSAM** outlines the strongest box (the fracture-tuned MedSAM for fractures, when on the computer).
- **Downloads blocked:** test mode never downloads weights.
  - CheXpert's weights were copied from vanda (the same public file the app already used there).
  - An automatic GitHub download attempt by torchxrayvision failed before that and saved nothing.
- **App button test 64/64** (3 new image checks: sorter + adult cut-off, chest model, head-CT gate).

**4. Pneumothorax** (`vision/app_cutoffs.py`, job 1533183; VinDr val 17 / test 15 pneumothoraces):

| Rule | Test caught | Normal X-rays unflagged |
|---|---|---|
| Detector box alone (old) | 53% | 99.5% |
| Detector OR CheXpert (cut-offs set together on val for >= 90% catch) | 86.7% | 45.9% |

Flagging half of all chest X-rays as "pneumothorax" would be alarm fatigue. So the app shows **two levels**:
- red = the detector's box;
- amber "possible pneumothorax, check the lung edges" = CheXpert only.

Detector + CheXpert by rank: AUROC 0.845 (detector 0.829, CheXpert 0.727). The real fix is more pneumothorax X-rays
(SIIM, 5o item 6).

**5. Urgency "95% catch" cut-off** (`outputs/metrics/urgency_cutoffs_95.json`; head + safety floor; test patients):

| Cut-off set on | Target | KTAS under-triage | KTAS non-urgent flagged | NHAMCS under-triage | NHAMCS non-urgent flagged | All patients flagged |
|---|---|---|---|---|---|---|
| KTAS val (app now) | 90% | 10.8% | 29.4% | 17.0% | 36.7% | 41-45% |
| KTAS + NHAMCS val | 90% | 8.1% | 43.8% | 9.4% | 49.6% | 53-57% |
| KTAS + NHAMCS val | 95% | 5.4% | 64.1% | 4.3% | 65.1% | 70-71% |

Reaching the 5% under-triage target means flagging 70% of all patients, far above the 25-35% over-triage the ACS
accepts. A cut-off cannot fix this; the model's discrimination is the limit. The app keeps the 90% (KTAS val)
cut-off, and the report should say so. More data (NHAMCS 2019-2021) is the next lever.

**3. Number reading**
- **Rules + AI cross-check** (`charting.read_numbers`): both readers read every note.
  - Agree → used.
  - Only one found it → used, marked with who read it.
  - Disagree → the rules' value is used until a nurse checks. The tile gets a dashed amber border and a "check" tag;
    Fix numbers settles it.
- **Rule fixes found while writing the group test cases:**
  - Ambulance-crew values come before "On arrival" values: the arrival values are now used, with a note saying so.
  - "BP 130 over 85" is now read.
  - "Brought by mum/dad/parents" counts as a walk-in.
- **Rule reader after the fixes:**
  - 100% on train and test, clean and messy, 0 wrong.
  - Same patient written 3 ways: 190/190 test patients read identically and correctly (`charting_check.py --consistency`).
- **Numbers adapter** (`make_number_data.py`: 6,770 train notes with exact quotes, more writing styles than the test
  template; `hpc/jobs/train_numbers.pbs`, job 1533190): results pending.

**Head-CT bleed model on plain grey pictures** (`app_cutoffs.py ich_grey`, job 1533183, RSNA test 30,126 slices /
2,558 patients):
- The RSNA JPGs pack 3 CT windows into the colour channels (brain, subdural, full range). A picture a clinician
  uploads is one grey slice.
- The grey copy was fine-tuned for 1 pass with half the slices turned grey; its cut-off was set on grey validation
  slices (0.5822).

| Model | Input | Slice AUROC | Patient AUROC | Patients caught | Patients unflagged (no bleed) |
|---|---|---|---|---|---|
| Original | colour windows | 0.973 | 0.988 | 97.1% | 89.9% |
| Original | grey | 0.849 | 0.904 | 81.9% | 84.4% |
| Grey copy (app) | colour windows | 0.973 | 0.987 | 96.8% | 89.7% |
| Grey copy (app) | grey | 0.968 | 0.985 | 97.3% | 84.9% |

**Head-CT checks in the group pack** (4 MultiCaRe pictures):
- Epidural and subdural bleeds flagged; the normal head CT not flagged.
- A chest CT forced through the model scored 0.94 "bleed", which is why it only runs when the notes say head.
- Its bleed-type guess called the epidural bleed "subdural" (epidural type AUROC 0.87), so the app writes "possible
  bleed" without a type.

**Group test pack** (`web/make_test_pack.py`, `web/guide_walkthrough.mjs`, `web/make_guide.py`; output
`web/test_pack/`, gitignored; zip 6.4 MB, guide 18 pages):
- Final run: all 30 checked cases pass, and the scripted screen walkthrough passed 18 of 18 steps.
- Steps covered: Submit, questions, answer, tiles, add info, fix numbers, chat, pictures, whiteboard, outcome +
  clear, view-only role.
- **40 cases:**
  - T01-T10: triage and number reading, including 3 wordings of one patient;
  - P01-P18: a note + a CC BY picture;
  - R01-R10: held-out real patients with their true level.
- **Checks:** every case was sent to the running test-mode app; all 30 checked cases pass.
- **Guide PDF:** screenshots come from a scripted headless-Chrome run of the real screens, which also checks each
  step.
- **What the pictures show** (left in on purpose):
  - P02: an adult fracture missed;
  - P03: a normal adult X-ray marked;
  - P10: a normal child's chest X-ray with a false shadowing flag;
  - P06: a pneumothorax caught (red), but with false effusion and shadowing flags on a journal figure.

### 5q. Hands-on test with the REAL AI, by mouse and keyboard (9 Nov 2026, 23:45-00:50, laptop)
Set-up: branch `ed-app-group-test`, `server.py` without `--test` on the laptop's RTX 3050 4 GB (MedGemma 4-bit + image,
task, questions and numbers adapters; questions/numbers adapters copied from vanda). Every step was done in Chrome by
clicking and typing (a second browser for the two-person whiteboard test). 6 patients + stress notes.

**Every button works** (as intended): Submit with name/empty-note checks, queue while busy, add a note while the AI
works, Open a text file, picture drop + Remove, live progress, Find a patient, choose a patient, click a question to
answer it, Fix numbers -> re-score (urgency 89% -> 97% after SBP 90 -> 82), Edit + Save ("Last edited by"), tick /
add / remove to-dos (case file and board stay in sync), stage, doctor, view-only Doctor role, leaving stage -> final
diagnosis + "Right level?" -> Clear -> Left the ED -> Put back, learning log (outcome and number fix recorded), Files
tab (as uploaded / AI marks / MedSAM outline + chips), Download the whole case file, Timeline, Notes as typed, Chat,
live whiteboard sync between two browsers.

**Accuracy with the real AI:**
| Patient | Numbers (rules + AI agree) | Level / urgent | Notes |
|---|---|---|---|
| T01 chest pain | 8/8 right | 2, urgent (expected 2) | case file lost PMH, meds, allergies, onset; 2 of 3 questions already answered in the note |
| T02 stroke | 6/6 right | 2, urgent, raised from 3 (expected 2) | onset "0830" lost; 1 fitting pathway to-do |
| P01 picture | 5/5 right | 4, not urgent (test mode 3) | fracture marked + outlined; MedGemma said "left ankle", and it IS an ankle (test-pack label was wrong, fixed) |
| impossible numbers | HR 400, RR 2, SpO2 150, T 45 dropped | 1, urgent | BP 300/20 accepted (drives level 1); each warning shown twice |
| T05 EMS vs arrival | 8/8 right, arrival values used | 2, urgent (expected 2) | both readers agreed |
| R03 real patient (NHAMCS) | 10/10 right, "brought by family" = not ambulance | 3, not urgent (true 5; test mode said 2, urgent) | its number step alone took ~11 min |

**Speed on the 4 GB laptop GPU** (MedGemma + 4 adapters do not fit: ~600 MB spills into shared memory):
- One patient took 4 to 17 minutes (T02 3.7 min, T01 6.7 min with a follow-up note, impossible-numbers note 10 min,
  P01 with a picture 17 min: reading the picture 5.5 min, writing the case file 7.5 min; T05 8 min).
- Numbers overall: 37 of 37 right across the 5 ordinary notes (plus the 4 impossible values dropped in the stress
  note); the rules and MedGemma agreed on every one,
  and the arrival-value rule worked (T05).
- Level and numbers appear after about 2 to 2.5 minutes; the case file and questions come later.
- **One lock for a whole patient run:** a nurse's "Save and re-score" waited 21 minutes and a chat question 34
  minutes behind other patients, with no "waiting" sign on Save.

**Problems found, most important first, and the fix plan:**
| # | Problem | Seen in | Fix | Effort |
|---|---|---|---|---|
| 1 | Case file drops history, past history, medicines, allergies: the arrival card's fixed "PAST HISTORY: not recorded" lines (`cards.py:77-80`) are in the record MedGemma reads, and it copies them over the note | real AI | leave the placeholder lines out of the record for the case file; afterwards file any note sentence still missing (word rules, `file_notes`) so nothing typed is lost | 1 h + re-run |
| 2 | Later notes filed under wrong headings (trop -> Presenting complaint, ECG -> History, aspirin -> Impression) | real AI | word rules first (`SORT_RULES`), MedGemma only for pieces the rules can't place | 1 h |
| 3 | Pathway flagger over-fires on very sick patients: 6 pathways (stroke, seizure, sepsis ...) for chest pain -> 10 to-dos, one duplicate | real AI | show at most the 2 strongest pathways; cut-offs for 95% specificity; merge to-dos with the same meaning | 2 h + re-test |
| 4 | Whole-run lock: Fix numbers / chat wait behind other patients; no waiting sign on Save | real AI | lock per AI step; re-score (one embedding, ~1 s) and chat jump the queue; "waiting for the AI (n ahead)" on Save | 2 h |
| 5 | Too slow on a 4 GB GPU | real AI | demo on a bigger GPU (vanda A40 or a Space with >= 16 GB); quick steps (numbers + level) for every queued patient first, writing afterwards | plan |
| 6 | Questions ask for what the note has; stroke question generic ("When did it start?") | real AI | mostly fixed by #1; keep the fixed pathway questions in real mode too | 30 min |
| 7 | Chat missed "since 0830" (note + nurse's edit) and gave a generic order | real AI | add arrival time and the notes as typed to the chat context | 30 min |
| 8 | BP 300/20 accepted; warnings shown twice | both | SBP 50-280, pulse-pressure check; de-duplicate warnings | 20 min |
| 9 | Whiteboard accepts edits with no name, and they are NOT on the timeline | both | same name check as Submit | 15 min |
| 10 | "Open a text file" accepts a JPG and fills the note with binary junk | both | accept .txt only; refuse if it isn't text | 10 min |
| 11 | Small: PDF on the picture box ignored silently; old red/green messages stay; "?, ?" before numbers; every Submit opens a new tab (6 tabs); BP tile still quotes "bp 90/60" after the fix to 82; board's learning line updates only on reload; board crowded by many to-dos; case-file edit is raw markdown | both | messages + one reused case-file tab + "fixed by" on tiles | 1-2 h |

**What this means for the group test:** the group runs test mode (no GPU), where #1-#7 don't appear because
MedGemma's writing is replaced; #8-#11 they can meet. **For the report and demo:** the decision support (numbers,
level, urgency, image models) is right in every real-AI case so far; the weak part is MedGemma's writing around it
(#1-#3, #6-#7), which is fixable without retraining.

### 5r. The 5q fixes, and how each was checked (9 Nov 2026, branch ed-app-group-test)
All in the shared back end, so the React app and the Gradio app both get them.

**Speed on the laptop GPU first** (scratchpad benchmark, same calls, same conditions; MedGemma 4-bit + 4 adapters):
| Step | adapters float32 (before) | adapters bfloat16 (now) |
|---|---|---|
| number reader (198 tokens) | 504 s | 246 s |
| case file, first 111 tokens | 451 s | 193 s |
| picture description | 89 s | 37 s |
| one picture checklist question | 76 s | 23 s |
The laptop stays far too slow for live use (under 1 token/s; it doesn't fit in 4 GB and spills into shared memory):
after the fixes, a text patient took 5.5-8 min (T01 466 s, T02 333-355 s), a patient with a picture 16 min (942 s) and a chat answer 3-9 min. On the vanda A40 the same pipeline
took a median of about 30 s per patient (5l).

| # | Fix | Checked |
|---|---|---|
| 1 | Record: the card's empty history lines point to the notes, and the arrival time is filled in (`visit.record`; the card the urgency model reads is unchanged). After MedGemma writes the case file, any piece of a note it left out is filed by the word rules, as typed (`charting.keep_every_note`; a clock time like 0830 must be there itself); a copied pointer line is removed | real AI: T01 now has "since 0600, sweaty", "hx DM, HTN on metformin", "NKDA", "BIBA"; T02 has "since 0830". Offline: a good case file is left unchanged |
| 2 | Later notes: the word rules file what they can; MedGemma only for the rest. New rule: onset / last known well -> History; "on <medicine>" only for drug-like words ("cut finger on glass" was past history) | real AI: trop + ECG -> Investigations, aspirin + cardiology -> Plan, no MedGemma call needed |
| 3 | Pathway flags: at most the 3 strongest (score over cut-off) + any named in the complaint; to-dos with the same meaning merged (`same_meaning`) | 1,452 test patients: caught 82.6% -> 80.6%, false flags per patient 0.76 -> 0.68, patients with 4+ flags 6.7% -> 0%. Real AI T01: 6 flags -> 3, to-dos 10 -> 5, no duplicate ECG |
| 4 | The model is held per AI step (`EDPipeline.gpu`), not for a whole patient; re-score and chat no longer queue behind other patients' jobs; "Re-scoring…" shown | code path; not timed under load |
| 5 | bfloat16 adapters (above); the picture checklist asked as one batch | batch gave the same answers; on the laptop no faster (memory-bound), expected faster on a real GPU but not measured. Not done: showing a level before the AI reads the note (a provisional level could mislead) |
| 6 | Pathway questions first in real mode too; "already answered" rules for onset, radiation, risk factors, mechanism of injury, witnessed collapse | real AI T02: first question "When was the patient last known to be well?"; test pack: 6 question rows changed, all dropped questions were answered in the note |
| 7 | Chat gets the time now, the arrival time and the clock times written in the notes ("since 0830") | before: "the time last known well is not stated". With the times written out it used 08:30, but worked the window out backwards ("last known well after 08:30 = outside"). So the app now also writes how long ago each time was ("since 0830, about 2 h 15 min ago"); not yet re-run with the real model. The chat stays advice for clinicians to check |
| 8 | BP: systolic 40-280, diastolic below systolic and at most 200 apart, never a diastolic alone; each warning once | BP 300/20 and 120/130 dropped, 240/120 kept; rules still 100% right, 0 wrong on train, test and messy test |
| 9 | Board and to-do edits need a name (server + board); board changes before the AI's first result are kept and put on the timeline | test_app: 2 new checks pass |
| 10 | "Open a text file" refuses non-text (React and Gradio) | by hand in both apps |
| 11 | Messages clear when typing; refused pictures say why; one reused case-file tab; "reading the note…" instead of "?, ?"; "fixed by <name>" on corrected tiles; learning line every 4 s | test_app, walkthrough |

**Tests after the fixes:** test_app 68/68 (4 new checks), test pack 40 cases all checks pass, guide walkthrough 18/18,
number reader 100% / 0 wrong.

**Gradio version** (`app/gradio/`, for a Hugging Face ZeroGPU Space, which only takes Gradio apps): Gradio
screens over the same `server.Store`; one patient's AI run = one `@spaces.GPU` call (patient copied in and back),
re-score and chat are calls of their own. Tested by hand in test mode, plain and with `BN4601A_FAKE_ZERO=1`
(copy-in/copy-out): submit, text file, name checks, case file, questions, to-dos, fix numbers, chat, board, clear,
left the ED, put back. Found and fixed on the way: an import race between two threads ("partially initialized module
polars"), a patient list that dropped patients who left the ED, a download button that saved nothing (it now always holds the current case file). The Space zip (`make_space_bundle.py`, 669 MB) was unzipped into an empty folder and run: picture, marks, MedSAM outline, edit and save all work. Not tried: a real ZeroGPU Space (4-bit loading there).

### 5s. The full AI without an NVIDIA GPU (10 Nov 2026, branch tidy-layout)

Testers without an NVIDIA GPU could only run test mode, because bitsandbytes' 4-bit loading was GPU-only.
`model.py` now picks the device itself (NVIDIA GPU -> Mac GPU -> processor). Tried on the laptop's processor
(Windows, 16 GB), same patient (67M chest pain, HR 112, BP 90/60), compared with the laptop's RTX 3050 (4-bit):

| MedGemma on the processor as | p(urgent) | level | one chat answer | case file of T01 |
|---|---|---|---|---|
| reference: GPU, 4-bit | 0.934 | 2 | 24 s | history, PMH (DM, HTN, metformin), NKDA, BIBA filed; 5.4 min a patient |
| bfloat16 (full precision) | 0.395 (other complaint wording) | model 3, final 2 | 197 s | history and PMH left "not recorded"; 28 min |
| 4-bit (bitsandbytes 0.50 runs on CPU) | 0.931 | 2 | 866 s | (too slow to run: ~2 h a patient) |
| **4-bit, unpacked to bfloat16 at load (`dequantize()`)** | **0.932** | **2** | **196 s** | **same as the GPU: history, PMH, NKDA, BIBA; 18 min** |

Lesson: the heads and adapters were trained on the 4-bit model's vectors, so a full-precision MedGemma shifts the
urgency score and the filing. Unpacking the 4-bit weights keeps exactly the values the heads saw (the chat answer was
word for word the GPU's) at full-precision speed. The questions for T01 were the same five on GPU and processor
(asthma/COPD, chest pain, witnessed collapse, fall injury, AVPU): some don't fit this note; the pathway flagger raises
breathing, collapse and head-injury pathways here, to look at after the group's feedback.

Layout change the same day (no behaviour change): the app moved to `app/` (server, frontend, Gradio, tests, tools,
build), Model 3's runtime stays in `reasoning/src`, training scripts in `reasoning/training`, step checks in
`reasoning/checks`. Checked after the move: 68/68 app checks (also from a fresh unzip), a 40-patient stress test
(0 errors, all on the board in order, 196 numbers read, 0 wrong), number reading 100% / 0 wrong on clean and messy
test notes, Caleb's 8 vitals tests, the React build (identical files), every Python file compiles, all vanda job
scripts parse.

## 6. Limitations (be honest)
- KTAS test has only 37 urgent patients, so its intervals are wide. NHAMCS (235 urgent) is the more
  reliable judge.
- Under-triage is 11–26% depending on dataset and target, well above the ACS < 5% benchmark. The score
  supports nurses; it doesn't replace them.
- Older patients score lower.
- The LoRA was trained on general medical chat, not on our triage format. Lower loss doesn't prove
  clinically better answers, and MedQA knowledge did not improve.
- Triage levels written by MedGemma (base or LoRA) miss 66–86% of urgent patients. They must not be used for
  urgency; the app uses the trained head + floor for that.
- The adapters hedge less and are less thorough than base MedGemma.
- Automatic answer checks (keywords, reading ease, hedging) are proxies, not a clinician's review.
- No clinician validated the outputs. Retrospective data only. Not for clinical use.

## 7. Engineering notes (useful for "what we learned")
- **Laptop RTX 3050 (4 GB):** 4-bit nf4 + double quantisation, everything on GPU 0.
- **Colab:** Gemma 3 can't use fp16 on a T4 (falls back to fp32), so it was too slow. Moved to the NUS HPC
  A40 (bf16).
- **HPC software:** the driver supports CUDA 12.4, so PyTorch is pinned to the cu126 build; Unsloth is
  installed alone so it picks matching libraries. Jobs go to the free routing queue `auto_free`.
- **Bugs found by checking results:** the MedQA scoring format, the bad training example, MedGemma's
  thinking tokens, the SBAR heading check, and the copy script silently failing.

## 8. References to cite
- MedGemma: Google, MedGemma technical report (2025); model card `google/medgemma-1.5-4b-it`.
- Fine-tuning and knowledge (verified 29 Oct 2026):
  - Zhou et al., "LIMA: Less Is More for Alignment", 2023, arXiv:2305.11206.
  - Biderman et al., "LoRA Learns Less and Forgets Less", TMLR 2024, arXiv:2405.09673. LoRA underperforms full
    fine-tuning on new domains, but forgets less.
  - Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?", EMNLP 2024,
    pp. 7765–7784, https://aclanthology.org/2024.emnlp-main.444. Examples with new knowledge are learned slowly
    and increase hallucination.
  - Hsieh et al., "Distilling Step-by-Step!", ACL 2023 (Findings), arXiv:2305.02301. Teacher rationales train
    small models well (basis for the task arm).
- Triage with LLMs:
  - "Knowledge-embedded large language models for emergency triage", Knowledge-Based Systems 2025,
    https://www.sciencedirect.com/science/article/abs/pii/S0950705125004782
  - "From Promising Capability to Pervasive Bias: Assessing LLMs for ED Triage", arXiv:2504.16273.
  - EQUITRIAGE, arXiv:2605.03998.
  - "Triage Performance Across LLMs, ChatGPT, and Untrained Doctors", JMIR 2024, PMID 38875696.
- LoRA: Hu et al., 2021, arXiv:2106.09685. QLoRA: Dettmers et al., 2023, arXiv:2305.14314. Unsloth (software).
  Hugging Face Transformers, PEFT, TRL.
- Layer choice: Skean et al., "Layer by Layer", ICML 2025, https://proceedings.mlr.press/v267/skean25a.html
- PromptEOL: Jiang et al., "Scaling Sentence Embeddings with LLMs", EMNLP Findings 2024, arXiv:2307.16645
- Decision curves: Vickers & Elkin, Med Decis Making 26(6):565–574, 2006.
- Triage benchmark: ACS Committee on Trauma under-triage < 5%, over-triage 25–35%
  (e.g. https://pmc.ncbi.nlm.nih.gov/articles/PMC7255776).
- Text + vitals at triage: Joseph et al., 2020, https://pubmed.ncbi.nlm.nih.gov/33145518/
- LLM acuity: Williams et al., JAMA Netw Open 2024, https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2818387
- NEWS2: Royal College of Physicians, 2017.
- Datasets: KTAS (Moon et al. 2019, PLoS ONE; Kaggle); NHAMCS 2022 (CDC NCHS); OpenMed SynthVision,
  MedDialog, Medical-Reasoning-SFT (Hugging Face); MedQA (Jin et al. 2021, `GBaker/MedQA-USMLE-4-options`);
  MultiCaRe (Nievas Offidani & Delrieux, 2024).
- Fracture detector: Abedeen et al., "FracAtlas: A Dataset for Fracture Classification, Localization and
  Segmentation of Musculoskeletal Radiographs", Scientific Data 10:521 (2023), doi:10.1038/s41597-023-02432-4.
  Ultralytics YOLO11 (software, Jocher et al.).
- Deployed-system comparison (5n): Levin S et al. Ann Emerg Med 2018;71:565-74 (e-triage). Raita Y et al. Crit Care
  2019;23:64. Ivanov O et al. J Emerg Nurs 2021 (KATE). FDA 510(k) K180647 and K221240 (Aidoc ICH); Voter AF et al.
  AJNR 2021 (Aidoc single-site). FDA 510(k) K212365 / K222176 (Gleamer BoneView); Kuo RYL et al. Radiology
  2022;304:50-62. FDA 510(k) K213941 (Annalise CXR pneumothorax). Lunit INSIGHT CXR, Clin Radiol 2023
  (doi 10.1016/j.crad.2022.08.129). Ambient scribes: NEJM AI 2025 randomised trial (DAX vs Nabla); Frontiers in AI
  2025 (doi 10.3389/frai.2025.1691499). FDA, Predetermined Change Control Plans for AI-enabled device software
  functions, final guidance, Dec 2024 (5m).
- Fix plan (5o, checked 8 Nov 2026):
  - Ma J et al. "Segment anything in medical images" (MedSAM), Nat Commun 2024;15:654.
  - Saporta A et al. "Benchmarking saliency methods for chest X-ray interpretation", Nat Mach Intell 2022;4:867-878.
  - Seyyed-Kalantari L et al. Nat Med 2021;27:2176-2182, doi 10.1038/s41591-021-01595-0.
  - Collins GS et al. TRIPOD+AI, BMJ 2024;385:e078378.
  - ELMTEX, arXiv:2502.05638 (2025). "For clinical data extraction, QLoRA attains accuracy close to LoRA",
    medRxiv 2025, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12633606/
  - Sclar M et al. "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design", ICLR 2024,
    arXiv:2310.11324.
  - Finlayson SG et al. "The Clinician and Dataset Shift in Artificial Intelligence", NEJM 2021;385:283-286.
  - Vasey B et al. DECIDE-AI, Nat Med 2022;28:924-933, doi 10.1038/s41591-022-01772-9.
  - Brooke J. SUS, 1996. Bangor A, Kortum P, Miller J. J Usability Studies 2009;4(3):114-123.
  - Kwong JCC et al. "The silent trial", Front Digit Health 2022;4:929508, doi 10.3389/fdgth.2022.929508.
  - Adam GA et al. "Hidden Risks of Machine Learning Applied to Healthcare: Unintended Feedback Loops...", MLHC
    2020, PMLR 126:710-731. Vaid A et al. Ann Intern Med 2023 (retraining after deployment can degrade models).
  - Goddard K, Roudsari A, Wyatt JC. "Automation bias: a systematic review", JAMIA 2012;19:121-127.
  - SIIM-ACR Pneumothorax Segmentation (Kaggle, 2019).
