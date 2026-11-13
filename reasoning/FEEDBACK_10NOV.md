# Group test feedback, 10-11 Nov 2026: issues and their causes

Sources (kept locally in `reasoning/outputs/feedback_10nov/`, not in git): Caleb's form (40 cases, main app, full AI,
NVIDIA 6 GB; from P04 on sent by a script), Cato's form (T01-P18, Gradio on Colab, full AI, plus general comments in
`cato.txt`), 58 screenshots, the group chat. Causes below were traced in the code or reproduced on 11 Nov
(`flag_scores.py` in that folder prints the pathway and urgency scores). Nothing is fixed yet.

## A. Urgency and level

| # | What testers saw | Cause (checked) |
|---|---|---|
| A1 | P05: 7-year-old, sore wrist, HR 96, flagged URGENT at 3% chance. T09 ("feeling unwell, no obs") flagged URGENT | `vitals/src/news2.py`: when 2 or more of RR / BP / HR are missing, the patient "could not be measured" -> HIGH risk -> safety floor 2 -> urgent. P05's note has no BP and no RR. Not the child's heart rate. The screen only says "safety floor from vital signs", so the chat then invented a reason ("due to HR 96") |
| A2 | Children get adult scores: P09 (4 y) NEWS2 10, P10 (5 y) NEWS2 3; a normal 7-year-old (HR 96, RR 22, BP 100/60) scores NEWS2 5 | NEWS2 is an adult score (16+), applied to every age. KTAS training data is 15+ (`ktas_data.py`), so the urgency and level models never saw children |
| A3 | R04, R05, R06 (true level 3) and R01 flagged urgent at 15-26% | Two things. (1) The cut-off: 0.124 at "catch 90% of urgent" (0.194 at 80%; only R05 would change). (2) The score depends on the AI-written complaint line: the same three patients with no complaint line score 5-7% (not flagged). So complaint wording moves the score by 10-20 points |
| A4 | Cato rated T04, P01-P03, P13, P15 "4 = bad": level differs from the guide | The answer key (`expected_results.csv`) was made in TEST mode, where the level is always 2 if flagged, else 3 (`test_backend.py`). The full AI's levels (4 for a cut finger) are the better ones. The key is wrong, not the app |

## B. Questions and to-dos

| # | What testers saw | Cause (checked) |
|---|---|---|
| B1 | Off-topic pathways: collapse + breathing for chest pain (T01); stroke for a headache (T07), sepsis (T03), effusion (P08); "possible poisoning" for head injuries (P11, P12) | The pathway heads flag anything over a cut-off set to catch 90% (`TARGET_SENS`), and the top 3 are shown. Scores / cut-off: T01 chest 5.4x, breathing 2.3x, collapse 1.6x, stroke 1.3x; T07 stroke 1.5x with the gate at 0.28 (cut 0.258); P11 poisoning 3.5x, stroke 2.8x (the word "warfarin" in the complaint); P12 poisoning 3.2x. The true pathway is far ahead in most cases (5.4x vs 1.6x), but anything >= 1.0x is shown as equal. Known cost: 0.68 false flags per patient on the test set |
| B2 | Each wrong flag fills the screen | Every flag adds 2-3 FIXED questions and a fixed to-do (`charting.PATHWAY_QUESTIONS`), fixed ones go first, and only 5 questions are shown (`server.py`), so two wrong flags push out MedGemma's own questions |
| B3 | "Trauma: mechanism of injury not recorded" on every injury (P01 "fell off ladder", P05 "fell from scooter") | `ed_assessment.py` adds that exact text whenever the patient is injured; pathway to-dos are not passed through `drop_known` (questions are) |
| B4 | "Any chest pain, fever or leg swelling?" for chest-pain patients; "on blood thinners?" for a 7- and a 9-year-old; "X-ray of the injured area" when the X-ray is already attached | Fixed question lists with no rule for the complaint itself, age, or attached pictures (`ANSWERED_BY_NOTE` has no entry) |
| B5 | T09: no observations at all, but only "level of consciousness?" asked | Nothing in the code asks for missing vital signs; it is left to MedGemma, which asked generic questions |

## C. Pictures

| # | What testers saw | Cause (checked) |
|---|---|---|
| C1 | P04 / P05: marks and outline drawn on a white picture; description "No evidence of fracture" next to a red fracture mark | Both child wrist pictures are 16-bit PNGs (mode I;16). `Image.open(path).convert("RGB")` clips them: 95-98% of pixels become white. The YOLO detector reads the file itself (so it still works), but the overlays, MedGemma's description and the checklist all get the white picture. Real X-rays are often 16-bit |
| C2 | Red marks on normal pictures: fracture on a normal adult leg (P03); pneumonia shadowing on a normal child chest (P10); cardiomegaly on a 4-year-old (P09) | Cut-offs chosen to miss little: adult fracture cut-off 0.0137 (marks 1 in 4 normal X-rays, as the guide says). The chest detector was trained on adults only (VinDr-CXR); a child's heart and thymus look big to it |
| C3 | Descriptions stop mid-sentence and run into the next text ("It's important to note that this is just a Head-CT bleed check:") | `describe_image` stops at 128 tokens and the detector's line is glued on with no cut at a sentence end |
| C4 | Odd checklist wording: "Joint effusion (a dash and in the ankle)"; left/right wrong (P02 note says R leg, description says left) | The checklist prompt says "yes, a dash and where it is" and MedGemma repeats the words "a dash and"; sides come from MedGemma (a 4B model) |
| C5 | Description and detector disagree (P09: "normal heart size" + red Cardiomegaly; P10: "bilateral infiltrates" on a normal film) | Two independent models, shown side by side with no note that they disagree |

## D. Chat and screens

| # | What testers saw | Cause |
|---|---|---|
| D1 | "Chatbot just regurgitates what the user inputs" | The chat prompt (`prompts.VISIT_CHAT_SYSTEM`) gets the whole record and no instruction to be brief; MedGemma restates the record before answering (see Cato's P05 chat) |
| D2 | "AI questions are broken: the AI can't answer the questions entered" | Design misunderstanding: these are questions FOR the nurse (click, type the answer, Add). The Gradio label "AI questions (click one to answer it)" reads as "the AI answers". Not checked on screen yet |
| D3 | Gradio whiteboard: urgent patients don't stand out; controls look like filters; no how-to | Plain `gr.Dataframe` and bare dropdowns; the React board has coloured level badges |
| D4 | Where to edit the case file isn't obvious; to-dos look like a checklist; wants one upload for text + pictures | Screen design (one small Edit button; checkbox list; two separate upload boxes) |
| D5 | T08 on Gradio: "HR 400 warning not shown" | Not reproduced. The code does print warnings under the numbers table as a plain bullet; probably missed because it isn't red. To check |

## What my first summary (11 Nov) got wrong
- P05 was blamed on "adult heart-rate rules for children". The real cause is the missing-vitals rule (A1). Children's scores are a separate problem (A2).
- "Move the cut-off to 80%" would un-flag only 1 of the 4 over-triaged real patients (A3).

## Fix order (most harm first)
1. C1 16-bit pictures (wrong AI text on real X-rays). 2. A1 say WHY the floor fired ("BP and RR not recorded: measure them") and use a softer flag for missing obs. 3. B1-B4 show only clearly-leading pathways; filter to-dos; age and complaint rules. 4. A2 / C2 label child patients "models trained on adults"; hide adult-only chest findings under 16. 5. B5 ask for missing obs. 6. C3-C5 text clean-up and "models disagree" note. 7. D1-D5 chat prompt and Gradio screens. 8. A4 remake the answer key with the full AI. A3 needs a decision (keep 90% and explain, or change).
