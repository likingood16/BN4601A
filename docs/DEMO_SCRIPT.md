# Live demo script: Mon 16 Nov, 11:00-12:30, E4-04-02 ALC

**Rules from the brief:** max **10 minutes**, max **3 slides**, mostly **live app**. Order: Background (why it
matters), Application (what we built), Live demo (show it works). Marks: Demo 15% (performance, presentation).

**The trick:** the full AI takes about 2 minutes per patient on the laptop. So the slow part happens **before**
you talk, or **while** the slides are on:
1. 15 min before: `demo_seed.py` fills the whiteboard with 6 patients.
2. The moment you start: submit T01 and P01, then do the slides (3 min) while the AI works.
3. By the time you switch to the app, both are ready.

---

## A. Day before (Sun): learn and rehearse (about 1.5 h)

| Learn | How | Time |
|---|---|---|
| Start the full AI | In `BN4601A\app`: `python server.py`, open `http://127.0.0.1:8000`. The top must say **Full AI** | 10 min |
| Fill the board | `python tools\demo_seed.py` (about 12-15 min, see C) | run once |
| Where every button is | Read `ED_app_user_guide.pdf` Part A (10 min), then click through T01 yourself | 30 min |
| The 6-minute demo path (section E) | Do it 3 times with a phone timer; aim for 5:30 | 40 min |
| The likely questions (section G) | Read them out loud once | 10 min |

## B. On the day: checklist (at 10:30)

- [ ] Laptop **plugged in**, Windows on "Best performance", sleep off, **Firefox and other heavy apps closed**
      (memory was 15/16 GB full on 13 Nov, which slows the AI)
- [ ] Wi-Fi not needed (everything runs on the laptop)
- [ ] `python server.py` running, browser at `http://127.0.0.1:8000`, name **Nurse Tan**, role **Triage nurse**
- [ ] `python tools\demo_seed.py` finished: the whiteboard shows 6 patients
- [ ] Two browser tabs: **Charting** (`#/`) and **Whiteboard** (`#/board`)
- [ ] T01 and P01 notes open in Notepad, ready to copy (`ED_app_test_pack\cases\T01.txt`, `P01.txt`), and the
      picture `pictures\adult_ankle_fracture.jpg` ready to drag
- [ ] Backup: `app\build\guide_shots\` open in Photos (if anything breaks, show the screenshots and keep talking)
- [ ] Slides open (Cato), presenter view
- [ ] Browser zoom 125% so the room can read it

## C. 15 minutes before: fill the board

```
cd BN4601A\app
python tools\demo_seed.py
```

It sends 6 background patients through the full AI, one after another (about 2 min each):
- T03 (sepsis);
- T04 (small cut);
- T06 (elderly confused);
- R05 (real patient);
- P06 (pneumothorax X-ray);
- P11 (head CT bleed).

It prints each one's level when it is done. Leave it running; don't type into the app while it works.

## D. Minute 0: submit, then slides (3 min)

1. In the **Charting** tab, paste **T01** and press **Submit**. A case-file window opens; leave it.
2. **New patient**, paste **P01**, drag in the ankle X-ray, **Submit**.
3. Switch to the slides:
   - **Slide 1, Why it matters (1 min):**
     - Triage nurses decide in minutes from messy notes.
     - Under-triage misses sick patients; over-triage clogs the department.
     - The AI suggests; the nurse decides.
   - **Slide 2, What we built (1 min):**
     - A messy note and pictures go in.
     - Numbers come out, each with the words it came from.
     - A trained urgency score + a vital-signs safety floor.
     - A case file, questions for the nurse, to-dos, a chat, and a whiteboard.
     - Three models: images (Cato), vital signs (Caleb), reasoning (Alex).
   - **Slide 3, Does it work (1 min):**
     - 40/40 test cases match the answer key.
     - 104 s per case on a laptop.
     - Trained vs base model.
     - A steadier urgency score.
     - A children's score.

## E. Live demo path (6 min)

| Time | Do | Say |
|---|---|---|
| 0:00 | Switch to the app. **Whiteboard**: 8 patients, most urgent first | "This is the ED whiteboard. Every patient, sorted by urgency, colour = triage level. The ✦ to-dos are suggested by the AI; a person ticks them." |
| 0:40 | Click **T01** (M, 67, chest pain). Point at the red **Urgent** flag and the tiles | "A messy note: '67M CP since 0600, HR 112, BP 90/60...'. The AI pulled out every number; under each one is the exact words it read it from. Urgent: 86-89% chance of level 1-2, and why: urgency score plus abnormal BP, HR and breathing." |
| 1:30 | Scroll the **case file** | "It wrote a tidy case file under standard ED headings. Nothing is invented: if it isn't in the note, it says not recorded." |
| 2:00 | Right side: **Questions for you to answer**. Click "What is the level of consciousness?": it appears in the **Add information** bar with "→". Type **alert, GCS 15** after the arrow, press **Add** (takes ~20 s) | "Clinical detective: the AI asks for the next most useful thing. I answer, and..." (wait) "...the consciousness tile fills in and it re-scores." |
| 2:45 | **Add information**: `trop 85 ng/L, ECG ST depression V4-6, aspirin 300 given`, **Add** | "New results come in, they go into the case file. Nothing already there is overwritten." |
| 3:15 | **Chat** tab: ask *Why is he urgent and what should we do first?* (~20 s) | "The chat reads the whole case file, including what we just added: troponin, ECG." |
| 3:45 | Ask: *Explain this to his wife in simple words*. Then the **SBAR handover** tab: **Write SBAR handover** (~30 s) | "Same case, explained for a non-specialist. And one click turns everything gathered so far, including the troponin and our questions, into an SBAR handover for the next doctor." |
| 4:15 | Open **P01** (ankle). **Files** tab: picture, AI marks, AI outline | "Pictures: a trained fracture detector draws the box, a segmentation model outlines it, MedGemma describes it. A mark means 'look here', not a diagnosis." |
| 5:00 | Back to the **Whiteboard**. On T01: set **Stage** to *Being seen*, **Doctor** *Dr Lim* | "The nurse and the organising doctor run the board. When a patient leaves, we record the final diagnosis, and the app learns from it." |
| 5:30 | Stop | "That's our ED triage assistant: it sees, reasons, communicates and assists, and a person always decides." |

## F. If something goes wrong

| Problem | Do |
|---|---|
| A patient is still "working" when you get to it | Show the whiteboard and P06/P11 (already done) first; come back |
| The app freezes or crashes | Say "let me show you what it produced" and open `app\build\guide_shots` (same screens, full AI) |
| Chat is slow | Talk over it: "It's MedGemma 4B on a 4 GB laptop GPU; on a real server it's seconds" |
| Laptop dies | Video (if done) or screenshots on a teammate's laptop |

## G. Likely questions

| Question | Answer |
|---|---|
| How accurate is it? | Urgency score AUROC 0.83 on 1,452 held-out patients from Korea and the US; the cut-off is set to catch 9 in 10 urgent patients, and on the held-out test patients it caught 86% (with the vital-signs safety rule). 40/40 test cases match the expected answers |
| Why does it over-triage? | On purpose: missing an urgent patient is worse. The cut-off catches 9 in 10, so some level-3 patients get flagged |
| What did training improve? | LoRA on MedGemma: shorter, always structured answers, same safety; retrained urgency score moves half as much when the same complaint is worded differently |
| Children? | A separate children's urgency score (trained on 1,256 children); the screen says which parts are still adult models |
| Is it a medical device? | No. Student prototype; a person always decides |
| Why MedGemma 4B? | Open weights, medical training, runs on a 4 GB laptop GPU in 4-bit |
