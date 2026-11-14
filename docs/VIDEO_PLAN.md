# Group video plan (today, Sat 14 Nov)

**Rules from the brief:** **MP4, max 3:00**, about **what the AI can do and why it matters**, show the **working
app, not slides**, no copyrighted media, peer-voted (a "most likes" contest). Marks: Video 15% (performance,
creativity, popularity). Upload to Canvas, Group Project folder, by Mon 16 Nov 23:59.

**Tools (already on your laptop, free):**
- **Snipping Tool** for screen recording;
- **Clipchamp** for editing, voice-over, captions and export;
- **Sound Recorder** for voice, if you'd rather record it separately.

**Always the real full AI** (`python server.py`, banner says "Full AI"). Speed the waiting parts up in the edit.

---

## 1. Plan for today (about 4 h)

| When | What | Time |
|---|---|---|
| 1 | Read section 2 (story) and practise the voice-over out loud twice | 20 min |
| 2 | Start the app (full AI), run `python tools\demo_seed.py` so the whiteboard looks busy | 15 min (runs by itself) |
| 3 | Record the 6 screen clips in section 3 (Snipping Tool) | 45 min |
| 4 | Record the voice-over, one take per scene (Clipchamp or Sound Recorder) | 30 min |
| 5 | Edit in Clipchamp (section 5): cut, speed up waits, put voice on top, captions, music | 90 min |
| 6 | Export 1080p MP4, watch it once on your phone, check ≤ 3:00, send to the group | 20 min |

> GPU note: only one full-AI app fits the laptop's graphics card. Another chat is using it for coding checks first;
> start recording once it says the GPU is free (or ask here).

---

## 2. Story and voice-over (2:50)

Tone: plain and factual, like describing a system. No drama. Short sentences. Small on-screen labels name the
brief's tasks so the markers can tick them off:
- **Core 1 Examine** = image detection;
- **Core 2 Explain** = chat;
- **Core 3 Report** = case file;
- **Creative task** = the AI asking for missing information, and the whiteboard that keeps re-ranking priority.

| Time | Scene (what's on screen) | On-screen label | Voice-over (about 140 words a minute) |
|---|---|---|---|
| 0:00-0:15 | Busy whiteboard | "ED triage assistant: decision support" | "This is a triage assistant for the emergency department. It supports the triage nurse and the doctor. The AI suggests; a person decides." |
| 0:15-0:35 | Charting: paste the messy T01 note, Submit | | "In most emergency visits, the first information available is a short typed note, the vital signs, and often an X-ray. The app is built around those three inputs." |
| 0:35-1:00 | T01: tiles with "read from the note's words", red Urgent flag | "Numbers: classification" | "First, the numbers. The vital signs are read from the note, and each one shows the words it came from. A classifier, trained on about 6,800 emergency visits from Korea and the US, gives the chance that the patient needs level 1 or 2 care. A vital-signs safety rule sits on top. This patient: 89 percent, urgent." |
| 1:00-1:25 | Case file, then "Questions for you to answer" → "alert, GCS 15" → Add → tile fills in | "Core 3 Report" then "Creative task: asks for what's missing" | "Second, the text. MedGemma, a transformer model built on Google's Gemma 3, which we fine-tuned with LoRA adapters, turns the note into a structured case file. It also lists what is missing, here the level of consciousness, to remind the professional what still needs checking. The answer is read back in, and the score is updated." |
| 1:25-1:50 | Chat: "Why is he urgent?", then "Explain to his wife in simple words"; then SBAR handover tab → Write SBAR handover | "Core 2 Explain" then "Core 3 Report: SBAR" | "The chat answers questions about the case using everything recorded so far, including new results such as a troponin, and can explain it in plain language for the family. One click then writes an SBAR handover from everything gathered." |
| 1:50-2:15 | P01 ankle X-ray: picture → AI marks → AI outline | "Core 1 Examine: object detection" | "Third, the images. In the emergency department, X-rays are the quickest imaging to get, including portable and handheld units at the bedside; CT and MRI take longer. So we use object detection: trained detectors mark fractures, collapsed lungs and brain bleeds, and a segmentation model outlines them. A mark means look here, not a diagnosis." |
| 2:15-2:40 | Whiteboard: patients in priority order; set stage and doctor | "Creative task: live priority" | "Every patient goes onto the whiteboard, ranked by priority. The ranking is re-evaluated whenever new information arrives, and the board shows the suggested to-dos for each patient." |
| 2:40-2:50 | End card: "40/40 test cases · under 2 min per patient on a laptop", names, "Student prototype, not a medical device", picture credits | | "On 40 test cases it matched the expected output every time, at under two minutes per patient on a laptop." |

Say it in your own words if a line feels stiff, but keep the facts the same.

**Facts in this script, and where they come from (in case someone asks):**
- 6,800 training visits: the KTAS (Korea) + NHAMCS 2022 (US) training split; see `reasoning/REPORT_NOTES.md`.
- 89%: T01 with the current urgency score.
- 40/40 and under 2 minutes: Caleb's 14 Nov retest, median 104 s.
- "X-rays quickest, CT and MRI take longer": general ED practice. If you want a number in the report, cite a source
  there; the video keeps it general on purpose.

---

## 3. Screen clips to record (Snipping Tool)

**How:**
1. Press **Win + Shift + R**.
2. Drag a box around the browser window. Use the same box every time: full browser, 1920×1080 if you can.
3. Press **Start**, do the action, press **Stop**.
4. **Save** (Ctrl + S) as `clip1.mp4` and so on, all into one folder, e.g. `Videos\ED_video`.

Turn the microphone **off** in Snipping Tool (you record voice separately). Browser zoom 125%.

| Clip | Record | Notes |
|---|---|---|
| clip1 | Whiteboard, slowly scroll down and up | after `demo_seed.py` has filled it |
| clip2 | Charting: paste T01, press Submit, then wait until the case file is done | the waiting is ~2 min: you'll speed it up 8x |
| clip3 | T01 case file: hover over the tiles, scroll the case file | slow mouse moves |
| clip4 | Click a question, type "alert, GCS 15", Add, wait, show the ACVPU tile filled | |
| clip5 | Chat: the two questions, wait for each answer; then the SBAR handover tab, Write SBAR handover | ~20 s each, SBAR ~30 s |
| clip6 | P01 (submit it earlier): Files tab, the three pictures | |
| clip7 | Whiteboard: set Stage "Being seen", Doctor "Dr Lim" | |

---

## 4. Record the voice-over

- **Clipchamp:** Home → **Create a new video** → left side **Record & create** → **Audio**. Record one scene
  at a time; it lands in your media.
- **Or Sound Recorder app:** one recording per scene, saved as `.m4a`; drag them into Clipchamp.
- Quiet room, phone earphones' mic is fine, 15 cm from your mouth, re-record any scene you stumble on (you'll cut
  the bad ones).

---

## 5. Edit in Clipchamp, step by step

1. **Create a new video**, aspect ratio **16:9**.
2. **Import media** (top left): all clips and voice files.
3. Drag clips onto the **timeline** in order (clip1 → clip7).
4. **Cut:**
   - click a clip on the timeline;
   - move the playhead to where you want to cut;
   - press **S** (or the scissors) to split;
   - select the unwanted piece and press **Delete**.
5. **Speed up the waiting:**
   - split around the waiting part;
   - select it;
   - top toolbar: **Speed** → 4x-8x;
   - add a small text "AI working… (sped up 8x)", which is honest and looks good.
6. **Voice:**
   - drag each voice file onto the **audio track** under the matching clip;
   - stretch or trim the clips so the picture matches the words.
7. **Zoom** (optional): select a clip → **Crop** / **Zoom** to make the tiles readable.
8. **Text:** left side **Text** → a simple title for scene 1 and the results line in scene 7.
9. **Captions:**
   - select the voice track;
   - right side **Captions** → **Turn on auto captions** (English);
   - fix any wrong words.

   Many people watch without sound.
10. **Music (optional):**
    - left side **Content library** → **Music** (royalty-free, allowed);
    - set the volume to about 10-15% so the voice stays clear.
11. **End card:** text slide with names, "Student prototype, not a medical device", and the picture credits
    "X-ray images: FracAtlas, GRAZPEDWRI-DX, MultiCaRe (CC BY 4.0)".
12. Check the length at the bottom right: **must be ≤ 3:00**.
13. **Export** (top right) → **1080p** → MP4. Name it e.g. `BN4601A_ED_triage_video.mp4`.

**Keyboard shortcuts:** Space (play/pause), S (split), Delete, Ctrl+Z (undo), Ctrl + scroll (timeline zoom).

---

## 6. Final checks before uploading

- [ ] MP4, ≤ 3:00, 1080p, plays on a phone
- [ ] Shows the real app ("Full AI" banner), not slides
- [ ] Voice clear over music; captions right
- [ ] No copyrighted music or pictures (Clipchamp library music is OK; credit the X-ray datasets)
- [ ] "Student prototype, not a medical device" shown
- [ ] Uploaded to Canvas, Group Project folder; link sent to the group
