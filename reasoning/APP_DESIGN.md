# ED app redesign: chart first, AI second (draft, 6 Nov 2026)

Status: **draft for the client (likingood16) to answer the open questions at the end**. Nothing built yet.

## 1. What the client wants (in their words, tidied)
- Nurses already know how to assess and take the numbers. **Our job is to format it**, not to make them fill a form.
- The nurse types **everything into one window**: messy notes, stats, comments. Images are uploaded in the same place.
- On **Submit**, the AI rebuilds the case file and a **separate window opens with the case file**.
- The AI **asks questions** (half the screen) about what's missing and what needs to be sent or done.
  A bar below the questions lets anyone **add information**.
- The case file is a **free-to-edit report**.
- **Chat** with the AI is in another panel, after submitting.
- Submitting also sends the patient to an **organisation whiteboard** on a **different screen**, showing what needs to be done.
  The organising doctor or triage nurse can change it.
- Build it as **React (front end) + Python (back end)**, not Gradio.

## 2. Why this direction is right (evidence)
| Finding | Source | What it means for us |
|---|---|---|
| ED doctors spent about 43% of a shift on data entry and only 28% with patients; about 4,000 clicks in a 10-hour shift (extrapolated, one hospital) | Hill et al. 2013, *Am J Emerg Med* 31:1591 | Forms and clicks are the problem. Typing freely and letting the AI format it removes clicks. |
| AI scribes in the ED save about 1–2 minutes of documentation per patient, but note quality is under-studied and some AI notes were judged worse than human ones | Stanford studies, *Ann Emerg Med* / *JMIR AI* 2026; Mayo pilot 2026 | Formatting saves time, but the clinician must be able to **check and edit** what the AI wrote. So the report must be editable, and AI text must be marked. |
| Electronic ED whiteboards act as the central information point for patient flow | Aronsky et al. 2008 (PMC2274795) | The whiteboard is a real ED tool, not a gimmick. It deserves its own screen. |
| Electronic whiteboards were read **less accurately at a distance** than dry-erase boards (font size, at-a-glance reading), though staff still preferred them | Rasmussen & Hertzum 2012 | Whiteboard: big fonts, few columns, colour for level, readable from across the room. |
| Simple redesigns removed most whiteboard inefficiencies in a keystroke-level analysis | Rasmussen & Kushniruk | Count clicks per task; aim for few. |

Design and PM books used for the structure below:
- Krug, *Don't Make Me Think* (3rd ed., 2014): one obvious action per screen; test with a few users early.
- Norman, *The Design of Everyday Things* (rev. 2013): visible status and feedback (the "has the AI finished?" problem).
- Nielsen, 10 usability heuristics (1994); Nielsen & Landauer 1993: about 5 test users find most problems.
- Fitzpatrick, *The Mom Test* (2013): ask about what users did last time, not "would you use this?".
- Cagan, *Inspired* (2nd ed., 2017): test value, usability and feasibility before building.
- MoSCoW prioritisation (DSDM) for the 10 days left.

## 3. Screens
**A. Charting screen (triage nurse, any device)**
```
┌──────────────────────────────┬───────────────────────────────┐
│ Patient: new / pick existing │ AI questions (after Submit)   │
│                              │ 1. Time of pain onset?        │
│  one big free-text box       │ 2. Any anticoagulants?        │
│  "67M CP since 0600, HR 112  │ 3. Send: ECG, troponin        │
│   BP 90/60 sats 94 RA, hx    │                               │
│   DM, on metformin..."       │ [answer / add info ......][+] │
│ [📎 images]        [Submit]  │                               │
└──────────────────────────────┴───────────────────────────────┘
```
**B. Case file window (opens automatically on Submit)**
- Top: urgency + level from the trained models, with the numbers the AI pulled out of the note (so a wrong number is visible).
- Body: the tidy report (standard ED note headings), **editable**. AI-written lines are marked until a person edits or confirms them.
- Below: the raw notes and images exactly as typed (never lost).
- Side panel: chat with the AI about this patient.

**C. Whiteboard (separate screen, e.g. a TV or second laptop)**
- One row per patient: level (colour), patient, complaint, stage, doctor, waiting time, **to do** (open items).
- Live: updates by itself when anyone submits.
- Organising doctor / triage nurse can change stage, doctor and tick off to-dos.

## 4. Flow
1. Nurse types the messy note (+ images) and presses **Submit**.
2. Back end, in order:
   1. AI pulls out the structured numbers (age, sex, vitals, complaint, arrival) → the existing urgency and level models.
   2. Images: sorted, read, checked (existing image tools).
   3. AI rebuilds the case file from all notes so far.
   4. AI writes questions: what's missing, and what needs to be sent or done (these become whiteboard to-dos).
3. Case file window opens; the whiteboard updates.
4. Answers and new info go back in through the bar → step 2 runs again for the new text only.

## 5. Build
- **Back end:** FastAPI (already installed, 0.141) wrapping the existing `pipeline.py`, `visit.py`, `image_tools.py`. One model in memory. Case files in SQLite (replaces the pickle). Live updates to the whiteboard with server-sent events.
- **Front end:** React + Vite. Needs Node.js on the laptop (**not installed yet**; about 30 MB download, ask first).
- **Test mode:** the fake model answers from `app/test_backend.py`, so the whole app can be clicked through without a GPU.
- **New AI prompts:** number extraction, case-file rebuild, questions/to-dos. All use the existing text adapter.
- **Safety:** numbers pulled by the AI are shown for checking; the trained urgency model still decides urgency; AI text is labelled.

## 6. Priorities for the 10 days (MoSCoW)
| Must | Should | Could | Won't (this project) |
|---|---|---|---|
| Charting screen + Submit | Images in the chart | Gradio/static mock of the whiteboard to agree the look first | Video |
| Number extraction → urgency | Live whiteboard on a second device | Voice input | Real logins / hospital systems |
| Case file window, editable | Tick-off to-dos | | |
| AI questions + add-info bar | Chat panel | | |
| Whiteboard screen | Usability test with 5 people | | |

## 7. Usability rubric (Poor / Acceptable / Good / Excellent)
"own" = a cut-off we chose ourselves, stated as such in the report.
| Criterion | How measured | Poor | Acceptable | Good | Excellent | Source |
|---|---|---|---|---|---|---|
| Overall usability | SUS questionnaire, 5 testers | < 50 | 50–67 | 68–79 | ≥ 80 | Bangor et al. 2008/2009; 68 = average |
| Effectiveness | % of 5 scripted tasks finished without help | < 60% | 60–79% | 80–94% | ≥ 95% | ISO 9241-11 (own cuts) |
| Efficiency: charting | Time from first keystroke to case file shown, one standard patient | > 5 min | 3–5 | 1.5–3 | < 1.5 | own; compare with the old Gradio app |
| Efficiency: clicks | Clicks to triage + submit one patient | > 15 | 9–15 | 5–8 | ≤ 4 | own; Hill 2013 for context |
| AI wait | Submit → case file on screen | > 90 s | 45–90 | 20–45 | < 20 | own |
| Whiteboard reading | Testers read level, stage and doctor correctly from about 3 m | < 70% | 70–84% | 85–94% | ≥ 95% | Rasmussen & Hertzum 2012 (own cuts) |
| Heuristic problems | Severity 3–4 problems left after a Nielsen-heuristic review | ≥ 3 | 2 | 1 | 0 | Nielsen 1994 |
| Faithful case file | Facts in the AI case file that are not in the notes | > 5% | 2–5% | 1–2% | < 1% | own; same check as image "invented details" |
| Useful AI questions | % of AI questions testers rate useful | < 40% | 40–59% | 60–79% | ≥ 80% | own |
| Number extraction | Vitals pulled correctly from 50 messy notes | < 90% | 90–95% | 95–98% | ≥ 99% | own (wrong numbers feed the urgency model) |

Five scripted tasks for testers: (1) chart a new chest-pain patient from a written vignette, (2) answer one AI question, (3) fix a wrong line in the case file, (4) on the whiteboard, assign a doctor and tick a to-do, (5) find which patient has waited longest.

## 8. Client answers (6 Nov 2026)
- Install Node.js: yes (v26.13.0 LTS, unpacked to `%LOCALAPPDATA%\Programs\node-v26.13.0-win-x64`, not on PATH).
- Gradio whiteboard mock first: yes (`app/whiteboard_mock.py`).
- Whiteboard edits: only the triage nurse and the organising doctor (name + role picked at the top; no real login).
- To-dos: the AI suggests them (marked ✦), and a person ticks them off or removes them.
- Whiteboard on a second laptop **and** in a second window: both work (one shared board on the server).
- After seeing the mockup: make the numbers easier to see (now big coloured tiles), add a Files tab, and check
  whether the AI detects or segments images based on the information.

## 9. What's built (6 Nov 2026, local only, not committed)
| Piece | File |
|---|---|
| Whiteboard mock (Gradio, made-up patients) | `app/whiteboard_mock.py` → http://127.0.0.1:7870 |
| Messy note → numbers (fixed rules), complaint, case file, questions (MedGemma or test rules) | `src/charting.py` |
| Back end (FastAPI): patients, AI job in a background thread, board, to-dos, chat, files, case-file download | `web/server.py` |
| Screens (React): charting `#/`, case file `#/case/<id>`, whiteboard `#/board` | `web/frontend/src/` |

**Images: detect → segment, matched to the note.**
1. MedGemma sorts the image type.
2. Cato's adapter describes the picture **blind**, without the note. Telling it what to expect can make it "see" it (anchoring).
3. Bone X-ray: the fracture detector marks possible fractures, and the strongest box goes to MedSAM automatically, so nobody has to click.
4. Chest X-ray: the 18-finding check runs.
5. The body parts the note mentions are written next to the reading, so the case file links the image to the complaint.

All three pictures (as uploaded, AI marks, AI outline) are shown in the Files tab.

On this laptop the detector works. MedSAM and the chest check are **not installed here** (they are on vanda). Until
they're downloaded, the real app says "outline not available" and test mode shades the box instead.

**Run it**
```
cd app/frontend && npm run build        # only after changing the screens (Node path above)
cd app && python server.py --test         # fake AI; add --lan for a second laptop on the same Wi-Fi
```
Open http://127.0.0.1:8000 (charting) and http://127.0.0.1:8000/#/board (whiteboard).
Without `--test` it loads MedGemma (needs the GPU).

**Checked in test mode:**
- Submit → numbers pulled out (HR 118 / BP 88/60 / RR 24 / SpO₂ 93 / pain 8, red or amber as NEWS2) → level 2 flagged.
- Case file, questions and to-dos appear; the whiteboard shows them.
- An X-ray gives detector marks plus an automatic outline in Files.
- Chat answers.
- A viewer's board change is refused (403).
- Videos are refused.

**Not checked yet:**
- the real MedGemma prompts (case file, questions, complaint);
- the new window opening in Chrome (the test browser pane doesn't open popups);
- a second laptop.

## 10. Dry run by Claude (6 Nov 2026, test mode): not a real usability test
Claude worked through the 5 scripted tasks in the browser, as a stand-in tester, to catch problems before real testers.
It gives no SUS score and doesn't stand in for the 5 human testers.

| Task | Result | Clicks | Problems found |
|---|---|---|---|
| 1. Chart a new patient from a vignette | done | 4 (+1 refused Submit) | name box far from Submit; example text looked like a real note; no "sent" message |
| 2. Answer / add information | done (Enter in the add bar) | 2 | none |
| 3. Fix a wrong line in the case file | done | 3 | header and whiteboard kept the old complaint |
| 4. Whiteboard: assign doctor, tick a to-do | done | 3 | to-dos were not buttons (no keyboard / screen reader) |
| 5. Find the longest-waiting patient | done | 0 | none (board sorts by level, waiting column readable) |
Also found in test mode: the note sorter cut "37.9" at the dot; "tightness" matched "htn"; "brought by ambulance" was taken
as the complaint; the tab title stayed "P1 · case file". **All fixed the same day.** Viewer role correctly can't edit.

**Number extraction** (`checks/charting_check.py`): KTAS patients written as messy notes in varied styles (BIBA, PR/pulse,
°F, GCS instead of AVPU, ...).
- **Dev run on train** found three problems: "BIBA" was not recognised; "97.0F" was read as age 0, female; and
  "no injury word" was being recorded as "no injury". All three were fixed.
- **Test, run once (190 patients):** vital numbers 100% correct, 0 wrong or invented. Age, sex, AVPU and arrival 100%.
- **Injury:** only 27% of injured patients are flagged from words (KTAS marks e.g. "chin pain" as injured). So injury is
  now left "not recorded" unless an injury word appears; it never claims "no injury".
- **Limit:** the notes come from our own templates, so real nurse notes (typos, other abbreviations) will score lower.

## 11. Whiteboard actions, stress test, every-button test (6 Nov 2026, test mode)
- **Board stages** (client's request): Waiting for doctor, Being seen, Results pending, then the four ways out: Going to
  surgery, Transfer to another department, Admitted to ward, Discharged home. Once a patient is on a way out,
  "✓ Clear from board" removes them from the board. The case file is kept under "Left the ED", and "Put back" undoes it.
- **Notes while the AI is busy** are now accepted, and the AI picks them up straight after. They used to be refused.
- **Later notes are only added** into the case file, under the right headings. A person's edits are never overwritten
  (the earlier version rewrote the whole file and lost an edit).
- **Stress test** (`web/stress_test.py`, test mode, 8 nurses at once): 190 KTAS + 200 NHAMCS + 60 MultiCaRe test patients.
  - **Load:** 0 errors, 450 patients done in 19 s, board answers in 35 ms with 450 rows.
  - **Numbers:** vital numbers 100% correct, none wrong or invented.
  - **Board:** every patient on it, most urgent first.
  - **Flagging:** test mode flags with Caleb's vitals model + safety floor only, not the MedGemma urgency head. KTAS
    sensitivity 0.89, NHAMCS 0.73. The real system's numbers come from the HPC run (`--direct --real`).
  - **Case reports:** vitals in MultiCaRe case reports were found in 33% of openings (they often appear later in the text).
- **Every-button test** (`web/test_app.py`): 56/56 functions do their job, including restart survival.
  - **Browser click check:** the charting screen (13/13) and the case file window both pass; the case-file failures
    found the lost-edit bug, which is now fixed.
- **Test notes** to load or paste: `web/test_notes/` (10 notes + answer_key.csv with the true level),
  made by `web/make_test_notes.py`.
