# Q&A prep: the story in order (demo day, 16 Nov)

**The one-line story:** messy note + vitals + X-ray → trained models decide the priority, with safety rules on top →
MedGemma explains, asks for what's missing and writes the case file → the doctor decides.

Sources for every number: `reasoning/REPORT_NOTES.md` (section numbers in brackets).

---

## 1. Data: what we used and why

| Data | What it is | Why we chose it |
|---|---|---|
| **KTAS** (Korea, Kaggle) | 1,262 adult ED visits with vitals, complaint and an **expert triage level** (KTAS 1-5) | Real triage labels from experts, with the vitals a nurse has at the desk. Caleb's vitals model and our urgency score are both trained on it |
| **NHAMCS 2022** (US CDC, public domain) | 8,412 adult ED visits (and 1,795 children) with vitals, reasons for visit, triage level, diagnoses | More data, and a **different country and triage system**, so a good test of whether the model travels. The diagnoses give us an answer key for pathways (stroke, chest pain…) |
| **MultiCaRe** (case reports, CC BY) | Real published ED cases with text and images | Realistic full-pipeline test notes and demo cases (Caleb's filter) |
| **Chat training mix** (SynthVision, MedDialog, OpenMed reasoning, MedQA) | 386k medical chat examples, tagged emergency or general | Teach MedGemma the style: short, structured, clinical answers |
| **Images** | **FracAtlas** (fractures), **GRAZPEDWRI-DX** (children's wrists), **Kermany** (children's chest X-rays), **VinDr-CXR** (chest findings), **RSNA** (head-CT bleeds), CheXpert weights | The ED's most common, fastest imaging: X-rays for fractures and chest, and head CT for bleeds. All open for academic use; the CC BY ones can be shown |

**Why these?**
- They have what triage really has at the start: a short note, vitals, and often an X-ray.
- They have labels we can test against: expert triage levels and final diagnoses.
- Same split every time (70/15/15, seed 42). The test patients were used **once**, at the end.

## 2. Base models: which and why

| Model | Why |
|---|---|
| **MedGemma 1.5 4B-it** (Google, built on Gemma 3) | Open weights; trained on medical text **and** images; instruction/chat; supports LoRA/QLoRA in transformers/PEFT/Unsloth. **Fits on a 4 GB laptop GPU in 4-bit**, so the app runs locally, with no patient data leaving the computer. The brief lists MedGemma as a suitable VLM |
| **YOLO11s-seg** (Ultralytics) | Fast object detection + segmentation; small enough for a laptop; the brief asks to visually ground findings |
| **MedSAM** | Outlines inside a detector's box (segmentation), so the mark shows the shape, not just a rectangle |
| **Caleb's vitals model** (NEWS2 + trained classifier) | Standard UK early-warning score, plus a model on vitals only: the safety floor |

## 3. Training: how and why

| What we trained | How | Why this way |
|---|---|---|
| **LoRA adapters on MedGemma** (Model 3 text) | 3 versions on the NUS HPC (A40 GPU): mixed, emergency-only, two-stage. Unsloth, 4-bit base, r = 16, alpha = 16, lr 1e-4. Plus a **task adapter**: 1 epoch on 8,274 teacher examples for triage reasoning, chat and SBAR (29 min on an A40) [5g] | LoRA trains about 1% of the weights, so it fits the GPU we had and doesn't wreck what MedGemma already knows. Compared **base vs adapted**, as the brief asks |
| **Urgency score** (classifier) | MedGemma turns the patient card into a vector (an embedding); a **logistic regression** on that vector predicts "needs level 1-2 care". Trained on KTAS + NHAMCS train; strength chosen on validation; cut-off set to **catch 9 in 10** urgent patients on validation; tested once | Few thousand labelled patients is too few to fine-tune a big model for a number, but plenty for a small classifier on good features. Gives a real **probability**, well calibrated (calibration error 2 points on NHAMCS, 7 on KTAS; Brier 0.12 / 0.08), measurable AUROC |
| **Steadier urgency score** (13 Nov) | Retrained with each training complaint **reworded by MedGemma** (nurse shorthand and plain words) [5u] | In the app MedGemma writes the complaint, so the wording varies; training on varied wording made the score move half as much |
| **Children's score** | Same method on 1,256 NHAMCS children [5u] | No child had been in the training data: on children the adult score reached only AUROC 0.70, and Caleb's adult vitals model was at chance (0.49) |
| **Triage level + "levels it can't rule out"** | Classifier on the same vector + conformal prediction | A level and an honest range, not false certainty |
| **Pathway flags** (stroke, chest pain…) | A gate + 7 small heads on the vector + complaint words; labels from final diagnoses | Catches "looks like a stroke" when the note doesn't say "stroke": reworded notes caught 80-100%, word rules 0% |
| **Image models** (Cato + ours) | Fracture detector (FracAtlas), chest detector (VinDr), head-CT bleed model (RSNA), picture sorter; Cato's LoRA for image descriptions | One specialist model per kind of picture beats one general model; the sorter routes each picture |

## 4. Software architecture: what and why

```
note + pictures → numbers read (MedGemma + rules, each with its source words)
                → patient card → MedGemma embedding → trained urgency / level / pathway models
                → + safety rules (vitals floor, missing vitals, critical picture finding): can only raise
                → priority on the whiteboard (re-checked whenever new information comes)
                → MedGemma (with adapters): case file, questions for what's missing, chat, SBAR
                → the doctor can lower or raise the priority
```

| Choice | Why |
|---|---|
| **Trained models decide, MedGemma explains** | Tested: MedGemma **alone under-triaged 66%** of urgent patients; guided by our score + floor, **23%** [rubric table]. The decision comes from a model we can measure; the LLM does language |
| **Safety rules only raise, never lower** | Under-triage is the dangerous error. Lowering is a doctor's decision |
| **Every number shows the words it came from; MedGemma and rules both read; disagreements go to the nurse** | No invented vital signs; the nurse can check at a glance |
| **One FastAPI back end, two front ends** (React for the ward, Gradio for the Hugging Face Space) | Same logic and tests for both; React is fast and tidy for daily use, Gradio runs on HF |
| **One model in memory, adapters swapped per step, one AI job at a time** | Fits a 4 GB GPU; a nurse's re-score waits seconds, not for another patient |
| **Test mode** | Fast fake writing, only for our 100+ automatic checks; never shown to users |
| **Learning from use** | Doctors' outcomes and corrections are logged; a retrained model goes live only if it does at least as well on the same held-out test patients |

## 5. Performance so far

| What | Result |
|---|---|
| Urgency score (1,452 held-out adults, Korea + US) | **AUROC 0.83** (KTAS 0.90, NHAMCS 0.82); with the safety floor catches **86%** of urgent patients (under-triage 14%), specificity 62%, PPV 34%, NPV 95%; Caleb's vitals-only model 0.78 / 0.61 on the same patients |
| Not one lucky split | 20 re-splits: KTAS AUROC mean 0.92 (0.85-0.97), NHAMCS 0.81 (0.80-0.83); beats vitals-only in 20 of 20 on both |
| Steadier score | Urgent flag changes with rewording: **15% → 8%**; reworded-test AUROC 0.79 → 0.82; same AUROC on the original notes |
| Children (270 held-out) | **0.79** vs 0.70 for the adult score |
| Triage level vs nurse | Weighted kappa **0.66** KTAS (good), 0.47 NHAMCS |
| LoRA, base vs adapted | Val loss 2.35 → **1.17**; SBAR **317 words vs 486** (shorter); SBAR uses chat facts 100%; reading ease 75 (fairly easy); MedQA unchanged (0.56 → 0.55): **style improved, not knowledge** |
| Image | Fracture detector (v3, in the app, own cut-off per age group): adults catches **83%**, 77% of normal X-rays unflagged (AUROC 0.88); children's wrists 82% / 99% (AUROC 0.99). MedSAM outline, fine-tuned: Dice **0.84** (0.48 before tuning, 0.39 box only); Cato's adapter findings F1 0.12 → 0.25 |
| Whole app | **40/40** answer-key cases match (Caleb's retest, **104 s** median per case on a 6 GB laptop GPU); in the full-AI answer key, 3 of 8 real patients exact and **none under-triaged** (R03-R06 over-triaged on purpose) |

## 6. Limitations (say these before they're asked)

- **Over-triage on purpose:** the cut-off catches 9 in 10, so some level-3 patients are flagged (R03-R06).
- **Image false marks** (P03, P07, P10): a mark means "look here", never a diagnosis; on adult X-rays 23% of normal ones get a mark.
- **Stroke to-do on some non-stroke cases:** kept because missing a stroke is worse; the retrained pathway model
  flagged more, so it was not used.
- **Labels are nurses' triage levels**, not patient outcomes; two countries, two triage systems.
- **Children:** small test set (26 urgent children); NEWS2 and the level are still adult models.
- **Speed:** about 2 minutes a patient on a laptop GPU; seconds on a server GPU.
- **LoRA didn't add medical knowledge** (MedQA flat); it changed style.
- **Single CT slices, no prospective test, not a medical device.**

---

## Tough questions, short answers

| Question | Answer |
|---|---|
| Why not let the LLM decide the triage level? | We measured it: alone it missed 66% of urgent patients. Trained score + safety floor: 23%, and our full app with the 9-in-10 cut-off catches 86% |
| What did fine-tuning actually improve? | Style and structure: shorter SBARs (317 vs 486 words), always the four headings, uses what was said in the chat. Not medical knowledge (MedQA flat), and we say so |
| How do you stop hallucinated numbers? | Every number must come with the exact words it was read from, and those words must be in the note; rules read too; disagreements go to the nurse |
| Why logistic regression, not a deep model? | A few thousand labelled patients: a small model on MedGemma's features generalises better, is calibrated and is easy to test |
| Is the test fair? | Same patients for old and new models, chosen before training, used once; external country (US) as well as Korea |
| Why X-rays and CT, not MRI? | X-rays are the quickest and most available ED imaging, including portable units; CT for head bleeds; MRI is rarely first-line in triage |
| What's your creative task? | Live priority from all three inputs (vitals, note, pictures) that re-checks whenever new information arrives, plus the AI asking for the most useful missing information |
| Who's responsible for a decision? | The nurse and the doctor. The AI suggests, explains and asks; only a doctor can lower a priority |
