# Try the ED app: 6 patients to paste in

Start it (laptop, test mode, no GPU needed):
```
conda activate bn4601a
cd "reasoning/ed_app"
python server.py --test
```
Open http://127.0.0.1:8000 for charting and http://127.0.0.1:8000/#/board for the whiteboard (a second window or a
second screen). Type your name top right and pick **Triage nurse** (only triage nurse and organising doctor can
change the board).

In test mode the numbers, rules, questions, to-dos and Caleb's vitals model are real; the AI *writing* is faked
(lines are filed by fixed rules instead of MedGemma, chat answers are placeholders). Real MedGemma: run it on vanda
(`hpc/upload.ps1`, then `hpc/connect.ps1`).

---

## 1. Chest pain, low BP (should be flagged urgent)
```
67M CP since 0600, sweaty, radiating to L arm. HR 112 BP 90/60 RR 22 sats 94% RA T 37.2 pain 8/10. hx DM, HTN on metformin. NKDA. BIBA
```
**While typing:** tiles fill in; BP 90/60 red, HR/RR/SpO₂ amber; "67 M · Ambulance".
**After Submit:** level 2, FLAGGED URGENT (vital-sign safety floor), NEWS2 8.
Red-flag line under the level: NEWS2 8 senior review, SBP ≤ 90 IV access (warnings, not to-dos).
To-dos (only things to send or do): ECG within 10 min, troponin.
"Missing" should be empty: onset, radiation, history, meds and allergies are all in the note.

**Then add (bar at the bottom):**
```
Trop 45 ng/L. ECG: ST depression V4-V6. Plan: aspirin 300mg, cardiology review
```
→ goes under Investigations and Plan. Edit the complaint line first (Edit → "Central chest pain since 06:00"), then
add the note: your edit must stay.

## 2. Possible stroke (pathway questions)
```
72F rt sided weakness and slurred speech. BP 182/100 HR 88 RR 18 sats 97% RA. walk-in
```
**Expect:** Missing → "When was the patient last known well?", "On blood thinners?". To-dos: capillary glucose, urgent CT
brain + stroke team. Press **Answer** on the first question and type `LKW 0830 today` → the question goes away and the answer is in
the case file.

## 3. Ambulance-crew numbers (must NOT be used)
```
61F lightheaded and flushing. EMS found BP 70/40 HR 130. On arrival BP 112/70 HR 92 sats 98% RA. alert. hx pulmonary hypertension on tadalafil
```
**Expect:** BP 112/70 and HR 92 on the tiles (arrival values), not 70/40 or 130. Delete the "On arrival …" sentence
before submitting and the tiles show a warning: BP is from before arrival, not used.

## 4. Head injury on a blood thinner
```
84M fell at home, hit head on table. on apixaban for AF. GCS 15. HR 78 BP 150/85 RR 16 sats 96% RA. pain 3/10. brought by family
```
**Expect:** fall / head-injury questions (loss of consciousness, how it happened) but NOT "on blood thinners?"
(apixaban is in the note). To-dos: CT head if NICE criteria met, X-ray of the injured area.
Try `no head injury` in a wrist-injury note instead: then there is no CT head to-do.

## 5. Fever, fast heart rate (sepsis to-do)
```
45F fever and rigors 2 days, dysuria. T 38.9 HR 118 BP 104/62 RR 24 sats 95% RA. confused. walk-in
```
**Expect:** AVPU "V" red (confused), temp/HR/RR amber–red. Red flags: altered consciousness → glucose and airway,
fever with fast heart rate → sepsis screen. To-dos: lactate, blood cultures, urine dipstick.

## 6. Wrist injury with an X-ray (pictures)
```
30M fell off bike onto L wrist, swollen and tender. pain 7/10. HR 84 BP 128/76. walk-in
```
Attach `reasoning/web/test_images/bone_xray_fracture_fracatlas.jpg` (button, drag, or paste). Under the picture,
**What is it?** lets you pick the type before submitting (bone X-ray → fracture check; chest X-ray → 18-finding
check), or leave **AI decides**.
**Expect:** case file → Imaging gets the reading; the **Pictures** tab shows the X-ray plus the fracture detector's marks.
In the Pictures tab:
- change **What is it?** → the picture is read again with that type's check (one reading kept);
- **Outline an area** → drag a box on the X-ray → an AI outline appears next to it;
- **✓ Reading is correct** → the Imaging line is marked checked.

---

## On the whiteboard
1. Patients are sorted most urgent first; the waiting time turns red after the level's target (L2 10 min, L3 30 min).
2. Set patient 1 to **Being seen** and type a doctor's name (Enter or click away saves it). The waiting clock stops
   and shows **seen after Xm**; patients still waiting keep counting.
3. Tick a ✦ to-do (click the chip).
4. Set patient 6 to **Discharged home** → **✓ Clear from board** → it moves to **Left the ED** (case file kept).
5. Switch your role to **Doctor**: the board becomes read-only.

## Things to try breaking
- Submit with no name → refused.
- Add a note while the AI is still working → accepted, added after.
- **Correct numbers** on a case file (e.g. SBP 82) → the level is scored again.
- Attach a PDF or video → refused.

Automatic version of all of this: `python test_app.py` (54 checks) while the server runs.
