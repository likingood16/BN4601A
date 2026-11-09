# How to test the ED app (every screen, every kind of input)

## Before you start
| What | Test mode (laptop, fake AI) | Real AI (vanda GPU) |
|---|---|---|
| Start | `cd reasoning\web` then `python server.py --test` | `powershell -File reasoning\hpc\app_on_vanda.ps1` once, then `powershell -File reasoning\hpc\connect_app.ps1` when the job runs |
| Open in Chrome | http://127.0.0.1:8000 (charting) and http://127.0.0.1:8000/#/board (whiteboard) | http://127.0.0.1:8010 and http://127.0.0.1:8010/#/board |
| AI answers | made up, marked TEST MODE | MedGemma 1.5 + task adapter + Cato's image adapter |
| Speed | about 1 s per patient | about 30-90 s per patient (first patient waits 2-4 min while the model loads) |

Put the whiteboard on a second screen, a second laptop (`--lan`), or just a second Chrome window.
At the top right of every screen, type your **name** and pick your **role**:
- **Triage nurse** or **Organising doctor**: can change the whiteboard.
- **Doctor**, **Nurse** or **Ward / other**: can only look at the whiteboard.

## Test inputs
- **Text notes:** `app/test_notes/`. There are 10 notes from held-out test patients, and `answer_key.csv` has the
  true triage level for each.
- **Images:** `app/test_images/`. There is one of each kind: bone X-ray with a fracture, chest X-ray (normal and
  pneumonia), CT, MRI, ultrasound, ECG and other. They are copied from vanda by `app_on_vanda.ps1`.
- **Video:** any .mp4. It should be **refused** with a reason (videos aren't read).

## Script (about 15 minutes); tick each box
### A. Charting screen (triage nurse)
1. [ ] **Open a text file:** press 📄 and pick `patient_02_nhamcs.txt`; the note fills the box.
2. [ ] **Submit:** you see "Sent", the case file opens in a new window, and the patient appears on the whiteboard.
3. [ ] **Check the level:** compare the board level with `answer_key.csv` (patient_02 is truly level 2, urgent).
4. [ ] **Questions:** the AI questions appear on the right. Click one, type the answer after the →, then press Add.
5. [ ] **New patient by typing:** type your own messy note, for example
   `72 yo man BIBA, SOB and chest tightness x2h. HR 124 BP 96/58 RR 26 sats 90% on air T 37.9, alert. hx heart failure on furosemide. allergic to penicillin`.
6. [ ] **Images, every kind:** pick **New patient**, type `45M fell off ladder, L wrist pain 7/10, HR 96 BP 130/80`, add
   `bone_xray_fracture_fracatlas.jpg` with 📎, then Submit. Then, using **Add to P…** in the patient menu, add each
   other picture in turn (chest, CT, MRI, ultrasound, ECG, other).
7. [ ] **Video:** add an .mp4. It must be refused.

### B. Case file window
8. [ ] **Header:** the level colour, urgency %, NEWS2, stage, doctor and waiting time are shown.
9. [ ] **Vital tiles:** big numbers, red or amber when abnormal. Hover a tile to see the words it came from.
10. [ ] **Fix numbers:** change HR, then Save and re-score. The level may change, and the Timeline logs it.
11. [ ] **Case file tab:** the tidy report. Press ✎ Edit, change a line, then Save; it says "Last edited by <you>".
12. [ ] **Add information:** type `trop 2.1, ECG ST depression V4-6` in the bottom bar and press Enter. The case file gains
    it, and **your edit from step 11 is still there**.
13. [ ] **Files tab:** for each picture you see as uploaded, the AI marks (bone X-ray), the AI outline (bone X-ray), and the
    chest X-ray finding scores. The AI picks the image type itself; check it got the type right.
    Click a picture to enlarge it. **⬇ Download the whole case file** gives one text file.
14. [ ] **Notes as typed** and **Timeline** tabs: everything is there, with who and when.
15. [ ] **Chat tab:** ask "What else should we rule out?". The answer uses the case file, including your edits.
16. [ ] **To do:** tick an AI to-do (✦) and add your own.

### C. Whiteboard
17. [ ] It shows the most urgent first, and the waiting time ticks. Readable from 3 m?
18. [ ] As **Doctor** (view only): you can't change anything.
19. [ ] As **Organising doctor**: set the Stage (Being seen, Results pending), set the Doctor, tick and remove to-dos, and add a to-do.
20. [ ] **Ways out:** set Going to surgery, Transfer to another department, Admitted to ward or Discharged home. The row
    sinks to the bottom. Then **✓ Clear from board**: the patient disappears, **Left the ED (n)** lists them, and
    **Put back** returns them.

### D. After each tester
- [ ] Questionnaire: the 10 SUS questions (APP_DESIGN.md section 7 has the scale).
- [ ] Note: tasks finished without help, time to chart one patient, clicks, anything confusing.

## Automatic checks (run any time)
- `python app/test_app.py`: every button's function, 56 checks (test mode).
- `python app/stress_test.py --url http://127.0.0.1:8001`: 450 test patients at once, checked against the board.
- `python reasoning/checks/charting_check.py --split test`: numbers pulled from messy notes, 190 KTAS patients.
