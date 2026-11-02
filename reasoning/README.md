# Model 3: Reasoning (MedGemma)

This folder is my part of the triage app. It uses **MedGemma** (`google/medgemma-1.5-4b-it`), a medical AI
model from Google, for two jobs:

1. **An urgency score.** How likely is this patient to be urgent (triage level 1–2)? It gives a number from 0 to 1.
2. **Writing.** It explains its triage reasoning, chats with staff, and writes an SBAR handover report.

Caleb's vitals model (`vitals/`) and his safety floor are reused here. The floor can only make a patient
*more* urgent, never less.

---

## The big picture

```
                        ┌──────────── JOB 1: URGENCY SCORE ────────────┐
patients (KTAS, NHAMCS) → patient card → MedGemma reads it → vector → head → p(urgent) → + Caleb's floor → flag
                           (cards.py)       (embed.py)                (heads.py)

                        ┌──────────── JOB 2: WRITING ───────────────────┐
chat examples (Pool 2) → LoRA training (lora_train.py) → adapter → compare with plain MedGemma (evaluate.py)
```

**Words used here**

| Word | Meaning |
|---|---|
| card | a patient written as short text: age, complaint, vitals, Caleb's vitals note |
| vector | 2,560 numbers MedGemma produces after reading a card: its "understanding" of the patient |
| head | a small, simple model (logistic regression) that turns the vector into p(urgent) |
| LoRA / adapter | a small add-on (~120 MB) trained on top of MedGemma. MedGemma itself is not changed |
| arm | one way of training LoRA: `mixed`, `emergency` or `two_stage` |
| AUROC | how well a score ranks urgent above non-urgent patients (0.5 = coin flip, 1.0 = perfect) |
| val loss | how surprised the model is by real answers it has never seen (lower = better) |

---

## Why each step exists

**1. Two data pools (`notebooks/02_datasets.ipynb`).** The two jobs need different data.
- **Pool 1 (patients):** KTAS (Korea, 1,262) + NHAMCS (USA, 8,412). Each patient has a true triage level, so we
  can train and test the score.
- **Pool 2 (chat examples):** 386k medical conversations (SynthVision, MedDialog, OpenMed reasoning, MedQA).
  These teach the writing style.
- **Split once, seed 42:** train / validation / test. **Test is used only once, at the very end**, so the result
  is honest.
- **Leakage guards:** no test patient or test question appears in training. KTAS test = Caleb's exact 190
  patients, so our numbers can be compared with his.

**2. Patient cards (`src/cards.py`).** MedGemma reads text, so each patient becomes a short card. Missing data is
written as "not recorded", so the model never guesses.

**3. Vectors (`src/embed.py`).** MedGemma reads each card once. We average its last layer into one vector per
patient. MedGemma is never retrained for this step.

**4. Score head (`src/heads.py`).** Logistic regression learns "which vectors mean urgent".
- Settings are picked on **validation**.
- The alarm threshold catches 80% of urgent patients on validation (Caleb's rule).
- Then Caleb's floor is added.

**5. LoRA training (`src/lora_train.py`).** It teaches MedGemma to write like the examples, in stages of
30,000 examples. After every stage it saves and measures progress. The three arms answer one question:
*is it better to train on everything, on emergency cases only, or general first and emergency second?*

**6. Comparison (`src/evaluate.py`).** Plain MedGemma vs each adapter, on the **same** held-out examples:
- val loss, split by source and by emergency vs general;
- MedQA exam accuracy on 200 held-out questions.

**7. Extra checks (`src/fusion.py`).** Does adding raw vitals to the head help? It also checks calibration,
decision curves, and how the score does for older vs younger patients and men vs women.

---

## Results so far

**Urgency score (test patients, never seen during training)**

| | KTAS AUROC | NHAMCS AUROC |
|---|---|---|
| MedGemma vector + head (trained on KTAS + NHAMCS) | **0.90** | **0.81** |
| Caleb's vitals model, same patients | 0.78 | 0.61 |
| Triage nurse (KTAS) | 0.92 | – |

- Adding raw vitals to the head: **no gain**. MedGemma already reads them from the card.
- The 4-bit laptop version scores the same as full precision. So the cheap version is enough for the app.
- Weak spot: older patients (65+) score lower on NHAMCS (0.77 vs 0.82).
- Caught at the 80% setting: 84% of urgent KTAS patients and 74% of urgent NHAMCS patients.

**LoRA (val loss, lower = better; MedQA out of 200 questions)**

| | all | emergency | general | MedQA |
|---|---|---|---|---|
| plain MedGemma | 2.35 | 2.35 | 2.35 | 56% |
| **mixed** (all 348k examples) | **1.17** | **1.16** | **1.17** | 53.5% |
| emergency only (74k) | 1.29 | 1.21 | 1.30 | 52.5% |
| two-stage (30k + 30k) | 1.27 | 1.24 | 1.27 | 56% |

- **Mixed wins everywhere**, even on emergency cases. More varied data beat focused data.
- MedQA did not improve. LoRA teaches *how to answer* (style, format), not new medical facts.
- MedQA is scored in each model's better format. Plain MedGemma likes to explain first; the adapters answer
  with the letter straight away.

---

## Folders

| Folder | What is in it | In git? |
|---|---|---|
| `notebooks/` | step-by-step notebooks for learning and checking (see `notebooks/README.md`) | yes |
| `src/` | the finished code: one file per step, each with a usage line at the top | yes |
| `hpc/` | scripts to run the heavy steps on the NUS HPC GPUs (see `hpc/README.md`) | yes |
| `models/` | trained adapters and heads (see `models/README.md`) | no, too big |
| `data/` | downloaded and processed datasets, made by notebook 02 | no, too big |
| `outputs/` | vectors, metrics (`outputs/metrics/*.json`), logs | no |

**Files in `src/`, in the order they are used**

| File | Job |
|---|---|
| `config.py` | all paths, model names and the seed in one place |
| `data.py` | load the tables notebook 02 made |
| `model.py` | load MedGemma (4-bit on the laptop, full on a big GPU) and `ask()` it something |
| `cards.py` | patients → cards |
| `embed.py` | cards → vectors |
| `heads.py` | vectors → urgency head + test results |
| `fusion.py` | Experiment 1: vector + vitals, plus the extra checks |
| `lora_train.py` | staged LoRA training (one arm) |
| `evaluate.py` | plain MedGemma vs adapters |

---

## How to run

**Laptop** (conda env `bn4601a`, from `reasoning/src`):

    python embed.py --table cards_pool1 --name base4bit      # vectors (about 2.5 h on the laptop)
    python heads.py --emb base4bit --train-on ktas+nhamcs    # head + test results
    python fusion.py --emb base4bit                          # Experiment 1 + extra checks

**NUS HPC** (LoRA training, full-precision vectors): see `hpc/README.md`.

## Still to do
Triage JSON prompt, chat + SBAR report, safety checks, `api.py` for the app, Hugging Face Space, report.
