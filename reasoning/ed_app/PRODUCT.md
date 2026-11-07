# ED Case Files v2: product spec

Status: built 7 Nov 2026 on branch `ed-app-v2`. Replaces `reasoning/web/` (kept for reference).
Run it: see "Run" at the end. The acceptance checks in section 5 are automated in `test_app.py`.

## 1. Problem
ED staff write quick, messy notes. The information then has to be re-typed into forms, follow the patient between
people, and be visible to whoever is organising the department. The app turns the messy note into a checked case
file and a shared whiteboard. **It supports clinicians' decisions and never replaces them.**

## 2. What was wrong with v1 (evidence: real-AI check run 3 on vanda, 7 Nov; dry run 6 Nov)
| # | Problem in v1 | Why it matters | v2 fix |
|---|---|---|---|
| 1 | Case file was one free-text block written by MedGemma; formatting differed per patient | Hard to scan; edits and AI text mixed | Case file = **items** under fixed headings, each with who wrote it and the words it came from |
| 2 | MedGemma wrote "SpO2 not recorded (room air assumed)" | An invented fact in a clinical record | Observations are **written by code** from the numbers; AI lines are **rejected by code** unless their quote is in the note and every number is in the quote |
| 3 | The EMS heart rate ("HR in the 80s" before arrival) was filed as the arrival HR and scored | Wrong number feeds the urgency model | Prehospital sentences detected; those values shown as "before arrival", **not scored** |
| 4 | Questions were the same for everyone (PMH / meds / allergies), to-dos repeated them ("Obtain PMH") | Noise; staff stop reading | Questions come from **ED pathways** for the complaint (stroke → time last known well). To-dos are **actions** (ECG ≤ 10 min). They never repeat each other |
| 5 | About 34 s per patient, 3–4 MedGemma calls (complaint, case file, questions) | Rubric "AI wait" only Acceptable | Two calls: the complaint comes out of the case-file call |
| 6 | Screens polled every 1–2.5 s; whiteboard rows overflowed | Load and lag; unreadable from 3 m | Server-sent events (push); board redesigned for a TV |
| 7 | Whole board re-written to a pickle on every change | A crash mid-write loses everything | SQLite, one row per patient |
| 8 | Numbers only visible after Submit | Wrong numbers found late | Tiles update **while typing** |

## 3. Users and jobs
| User | Job | Screen |
|---|---|---|
| Triage nurse | Type everything once, attach pictures, check the numbers, submit | New patient |
| Triage nurse / organising doctor | Run the board: stage, doctor, to-dos, way out | Whiteboard (edit) |
| ED doctor, other nurses, ward | Read the case file, check AI lines, add results, answer questions, ask the AI | Case file |
| Everyone else | See the board | Whiteboard (read only) |

## 4. Requirements (MoSCoW)
**Must**
- M1 One free-text box + pictures; Submit opens the case file in a new window and adds the patient to the board.
- M2 Numbers (age, sex, HR, BP, RR, SpO₂, temp, pain, AVPU, arrival, injury) by fixed rules, as colour tiles (NEWS2 bands) with the words they came from.
- M3 Urgency + level from the trained models (MedGemma-embedding heads + Caleb's safety floor); the triage section is their decision word for word.
- M4 Case file under standard ED headings; uses only facts in the notes (checked by code); AI lines marked until a person checks them.
- M5 Questions about missing information and suggested to-dos, specific to the complaint.
- M6 Add information at any time, even while the AI is busy; it is filed under the right headings; people's edits are never overwritten.
- M7 Whiteboard: one row per patient, most urgent first, level colour, complaint, stage, doctor, waiting time, to-dos (✦ suggestions, ticked by a person). Only the triage nurse and organising doctor can change it.
- M8 Stages: Waiting for doctor, Being seen, Results pending, then the ways out (surgery, transfer, ward, home). Patients on a way out can be cleared; their file stays under "Left the ED".

**Should**: pictures (MedGemma sorts the type; bone X-ray → fracture detector → MedSAM outline; chest X-ray → 18-finding check); chat that reads the latest case file; correct numbers → re-score; download case file; timeline of who did what.

**Won't (this project)**: real logins, hospital system integration, video, voice.

## 5. Acceptance checks (all automated in `test_app.py`)
| Area | Check |
|---|---|
| Numbers | Arrival value used, EMS value ignored; EMS-only value shown as "before arrival" and not scored; all 9 values right on the standard chest-pain note |
| Faithful | Observations by code, room air recorded, nothing "assumed"; every AI line's quote is in the note; every fact typed appears in the case file |
| Decision | Triage section = models' decision; SBP 90 → safety floor flags; every truly urgent test note flagged |
| Questions | Stroke → time last known well; no question asks for what the note has; answering removes the question and files the answer; dismissed questions don't come back |
| To-dos | Chest pain → ECG; stroke → glucose + CT; no "Obtain …" to-dos; a removed AI to-do isn't suggested again |
| Edits | A person's edit survives later notes; notes accepted while the AI is busy; new results go under the right headings; lines can be checked and added |
| Numbers fixed | Correcting numbers re-scores; observations rewritten without duplicates; later obs offer a re-score |
| Board | Most urgent first; a doctor (viewer) gets 403 on board and to-dos; can't clear a patient who hasn't left; cleared → "Left the ED" with the file kept; to-do ticks and timeline record who |
| Other | Chat answers (real AI: reads the latest file); export has every heading; X-ray read and marked; non-pictures and unsigned notes refused |
| Speed (real AI) | Submit → case file ≤ 45 s (rubric "Good") |

## 6. Design rules
- **AI suggests, people decide.** Urgency comes from validated models, not the language model. Every AI line is marked ✦ until checked.
- **Nothing typed is lost.** Any piece of a note that no checked AI line covers is filed in the nurse's own words.
- **Nothing invented.** Code checks every AI line against the note; rejected lines are listed, not hidden.
- **One obvious action per screen** (Krug): Submit; check/edit; answer; tick.
- **Readable from 3 m** (Rasmussen & Hertzum 2012): big type, colour + number for level, few columns.

## 7. Files
| File | Job |
|---|---|
| `server.py` | FastAPI back end: patients, AI worker queue, SQLite, live events, roles |
| `extract.py` | Note → numbers with source words, prehospital detection |
| `casefile.py` | Case-file items, AI filing + checking, code-written sections |
| `questions.py` | ED pathways, vital-sign red flags, MedGemma extras |
| `ui/src/` | React screens: `Chart.jsx` (new patient), `Case.jsx` (case file), `Board.jsx` (whiteboard) |
| `test_app.py` | End-to-end acceptance test (section 5) |
| `hpc/` | Run on vanda with the real AI: `upload.ps1`, `connect.ps1`, `ed_app.pbs` |

## 8. Run
```
# laptop, test mode (fake writing; rules + Caleb's vitals model real):
conda activate bn4601a
cd reasoning/ed_app && python server.py --test           # http://127.0.0.1:8000  and  /#/board
python test_app.py                                          # acceptance checks
# screens changed? (Node at %LOCALAPPDATA%\Programs\node-v26.13.0-win-x64)
cd ui && npm run build
# real AI on vanda (NUS VPN on):
powershell -ExecutionPolicy Bypass -File reasoning\ed_app\hpc\upload.ps1
powershell -ExecutionPolicy Bypass -File reasoning\ed_app\hpc\connect.ps1   # http://127.0.0.1:8011
```
