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

## 5f. App design: a triage desk, checked against real practice (`reasoning/app/`, 2 Nov 2026)
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

## 5h. Rubric: every model graded on its own held-out test (`src/rubric.py`, 6 Nov 2026)
Reads saved results only; re-run `python src/rubric.py` after new results. Grades Poor / Acceptable / Good / Excellent;
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
