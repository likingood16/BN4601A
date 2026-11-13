"""The user guide PDF for the group test (8 Nov 2026), built from the same run as the test pack.

    python make_test_pack.py --url <full-AI app>   # 1. runs every case through the FULL AI -> expected_results.csv
    node guide_walkthrough.mjs <fresh full-AI app> ../build/guide_shots ../build/ED_app_test_pack
                                        # 2. clicks through it in headless Chrome -> screenshots + checks (1-2 h on a laptop)
    python make_guide.py                # 3. this: HTML from the screenshots + answer key -> ED_app_user_guide.pdf,
                                        #    copied into the pack, and the pack zipped again

Everything the guide says the app does is either a screenshot of it doing it, or a row of expected_results.csv that
make_test_pack.py checked against the running app.
"""
import csv
import html
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "build"
OUT = PACK / "ED_app_test_pack"
SHOTS = PACK / "guide_shots"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
E = html.escape


def fig(name, caption, width=100):
    return (f'<figure style="width:{width}%"><img src="guide_shots/{name}.png"><figcaption>{caption}</figcaption></figure>')


def rows_table(rows, prefix):
    out = ['<table class="key"><tr><th>Case</th><th>What it tests</th><th>What the app showed (full AI)</th></tr>']
    for r in rows:
        if not r["case"].startswith(prefix):
            continue
        showed = f"Level {r['level']}, {'URGENT' if r['flagged_urgent'] == 'yes' else 'not flagged'}"
        if r.get("picture_sorted_as"):
            showed += f"; sorted as {r['picture_sorted_as'].replace('_', ' ')}; {r['picture_findings']}"
        if prefix == "T":
            showed += f"; read: {r['numbers'].replace('sbp ', 'BP ').replace(', dbp ', '/')}"
        if r["number_warnings"]:
            showed += f"; note: {r['number_warnings']}"
        if prefix == "R":
            showed += f" <span class='muted'>({E(r['what_the_guide_says'].split(';')[0])})</span>"
            out.append(f"<tr><td>{r['case']}</td><td>{E(r['title'])}</td><td>{showed}</td></tr>")
            continue
        out.append(f"<tr><td>{r['case']}</td><td><b>{E(r['title'])}</b><br><span class='muted'>Look for: "
                   f"{E(r['what_the_guide_says'])}</span></td><td>{E(showed)}</td></tr>")
    return "\n".join(out) + "</table>"


def trim(png):
    """Cut the empty page background under a screenshot (the screens are shorter than the window)."""
    import numpy as np
    from PIL import Image
    img = Image.open(png).convert("RGB")
    a = np.asarray(img).astype(int)
    bg = a[-1, 0]
    busy = (np.abs(a - bg).sum(2) > 24).mean(1) > 0.01
    last = int(np.nonzero(busy)[0].max()) if busy.any() else a.shape[0] - 1
    if last < a.shape[0] - 80:
        img.crop((0, 0, a.shape[1], last + 40)).save(png)


def main():
    for png in SHOTS.glob("*.png"):
        trim(png)
    rows = list(csv.DictReader(open(OUT / "expected_results.csv", encoding="utf-8-sig")))
    checks = (SHOTS / "walkthrough_checks.txt").read_text().splitlines()
    n_real = sum(r["case"].startswith("R") for r in rows)
    agree = sum(r["case"].startswith("R") and r["what_the_guide_says"].startswith(f"True triage level {r['level']} ")
                for r in rows)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>ED triage app: user guide and test plan</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; color: #1d2733; line-height: 1.45; }}
h1 {{ font-size: 24pt; margin: 0 0 4px; color: #0d47a1; }}
h2 {{ font-size: 15pt; color: #0d47a1; border-bottom: 2px solid #bbdefb; padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }}
h3 {{ font-size: 12pt; margin: 14px 0 4px; page-break-after: avoid; }}
figure {{ margin: 8px 0 12px; page-break-inside: avoid; }}
figure img {{ width: 100%; border: 1px solid #cfd8dc; border-radius: 6px; }}
figcaption {{ font-size: 9pt; color: #546e7a; margin-top: 3px; }}
.box {{ background: #e3f2fd; border-left: 4px solid #1565c0; padding: 8px 12px; margin: 10px 0; page-break-inside: avoid; }}
.warn {{ background: #fff3e0; border-left: 4px solid #ef6c00; padding: 8px 12px; margin: 10px 0; page-break-inside: avoid; }}
code, .mono {{ font-family: Consolas, monospace; background: #f1f4f7; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }}
pre {{ font-family: Consolas, monospace; background: #f1f4f7; padding: 8px; border-radius: 4px; font-size: 9pt; white-space: pre-wrap; }}
table {{ border-collapse: collapse; width: 100%; margin: 6px 0 12px; font-size: 9pt; }}
th, td {{ border: 1px solid #cfd8dc; padding: 4px 6px; vertical-align: top; text-align: left; }}
th {{ background: #eceff1; }}
tr {{ page-break-inside: avoid; }}
.muted {{ color: #607d8b; }}
.chip {{ display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 9pt; }}
.red {{ background: #fdecea; color: #c62828; font-weight: 600; }} .amber {{ background: #fff3e0; color: #a15c00; }}
.green {{ background: #eef5ee; color: #2e7d32; }}
.steps li {{ margin-bottom: 4px; }}
.page {{ page-break-before: always; }}
.two {{ display: flex; gap: 12px; }} .two > * {{ flex: 1; }}
</style></head><body>

<h1>ED triage app</h1>
<div style="font-size:13pt;color:#455a64">User guide and group test plan · BN4601A, Model 3 · version of {time.strftime("%d %b %Y")}</div>
<div class="warn"><b>Student research prototype.</b> Not a medical device. Never use it with real patients' details.
All test notes and pictures in this pack are from public datasets or written for testing.</div>

<h2>What the app does, in one minute</h2>
<p>A triage nurse types the arrival note the way they normally would (shorthand is fine) and can add X-rays or scans.
When they press <b>Submit</b>, the app:</p>
<ol class="steps">
<li>pulls the numbers out of the note (heart rate, blood pressure and so on) and shows the words each number came from;</li>
<li>scores how urgent the patient is with the trained models and a vital-sign safety floor, and gives a triage level (1 = most urgent);</li>
<li>writes a tidy case file from the note, asks about what is missing, and suggests to-dos;</li>
<li>sorts each picture (bone X-ray, chest X-ray, CT, ...) and runs the trained model for that kind, which marks where to look;</li>
<li>puts the patient on the <b>whiteboard</b>, the shared list of everyone in the ED.</li>
</ol>
<p>Everything the AI adds is marked as AI (✦), and a person always makes the decision.</p>

<h2>1. Install it and start the full AI (about 20 minutes, plus an 8.6 GB download the first time)</h2>
<ol class="steps">
<li><b>Download</b> these files from the group chat into <b>one folder</b> (e.g. <code>Downloads\\EDapp</code>; Telegram
Desktop's own <code>Downloads\\Telegram Desktop</code> also works): <code>ED_app_code.zip</code>,
<code>ED_app_full_ai_models.zip</code> (715 MB: the trained models; keep it in the group) and
<code>ED_app_test_pack.zip</code> (test cases, pictures, answer key, feedback form).</li>
<li><b>Python 3.11:</b> if you don't have Anaconda or Miniconda, install Miniconda (docs.conda.io/en/latest/miniconda.html).
Open <b>Anaconda Prompt</b> (Windows) or <b>Terminal</b> (Mac).</li>
<li><b>Unzip</b> <code>ED_app_code.zip</code> (right-click → Extract All) and <code>ED_app_test_pack.zip</code>.</li>
<li><b>Set up and start</b>, one line at a time (change the first line to where your <code>BN4601A</code> folder is; tip: type
<code>cd </code> and drag the <code>app</code> folder into the window):</li>
</ol>
<pre>cd Downloads\\EDapp\\ED_app_code\\BN4601A\\app
conda create -n edapp python=3.11 -y
conda activate edapp
pip install -r requirements.txt              (5-10 minutes, only the first time)
python install_models.py                     (finds ED_app_full_ai_models.zip, should end with "all there. Full AI")
python server.py                             (the full AI; the first start downloads MedGemma, 8.6 GB)</pre>
<p><b>NVIDIA GPU (much faster):</b> before <code>python server.py</code>, also run<br>
<code>pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall</code> and
<code>pip install bitsandbytes</code>.</p>
<p>Open <b>http://127.0.0.1:8000</b> in Chrome or Edge when it says <b>Uvicorn running</b>. The whiteboard is at
<code>http://127.0.0.1:8000/#/board</code>. <b>Next time:</b> <code>conda activate edapp</code>, the <code>cd</code> line, then
<code>python server.py</code>. <b>Problems:</b> "port 8000 in use" → <code>python server.py --port 8001</code>;
"python is not recognised" → use Anaconda Prompt; it closes while loading → not enough free memory (close other apps).
<b>The Gradio version</b> of the app uses the same install: see its own guide, <code>ED_app_gradio_guide.pdf</code>.</p>
<div class="box"><b>First thing every time:</b> type <b>your name</b> top right and pick your <b>role</b>. Without a name,
Submit is refused ("Type your name next to Submit"). Only the <b>Triage nurse</b> and <b>Organising doctor</b> can
change the whiteboard and tick to-dos; other roles can look.</div>

<h2>1b. How fast it is on your computer</h2>
<p>The full AI runs on any computer with <b>16 GB of memory</b>; it picks the fastest place by itself.</p>
<table>
<tr><th>Your computer</th><th>Where MedGemma runs</th><th>Speed (measured 10 Nov unless marked)</th></tr>
<tr><td>NVIDIA GPU, 8 GB or more</td><td>the GPU (4-bit)</td><td>about 1-3 minutes a patient (30 s on a 48 GB GPU)</td></tr>
<tr><td>NVIDIA GPU, 4-6 GB</td><td>the GPU (4-bit)</td><td>about 5 minutes a patient (case T01: 5.4 min on a 4 GB RTX 3050)</td></tr>
<tr><td>Mac with an M1-M4 chip</td><td>the Mac's GPU</td><td>not measured yet: expect 5-15 minutes a patient</td></tr>
<tr><td>Any other laptop (Intel / AMD, no NVIDIA)</td><td>the processor (the same 4-bit model, unpacked)</td><td>starts in
about 2 minutes; case T01 took 18 minutes (urgency and level after 6 min, the case file after 14); a chat answer about
3 minutes. Same scores and wording as on a GPU</td></tr>
</table>
<p>The screens keep working while the AI writes: the numbers, urgency and level appear first, then the case file,
questions and to-dos. On a slow computer, try two or three patients and the chat rather than every test case.</p>

<div class="page"></div>
<h2>Part A. One patient from start to finish (case T01)</h2>
<h3>A1. Write the note</h3>
<p>On <b>Charting</b>, leave <b>Patient</b> on <b>➕ New patient</b>. Type or paste the note, or press
<b>📄 Open a text file</b> and pick <span class="mono">cases/T01.txt</span>. Pictures can be dropped on the dashed box.</p>
{fig("02_charting_typed", "The charting screen with case T01 typed in. Shorthand such as CP, BIBA, NKDA and RA is fine.")}
<h3>A2. Submit</h3>
<p>Press <b>Submit</b>. A green line says it was sent; <b>the case file opens in the case-file tab</b> (the same tab for every patient) and the patient appears on
the whiteboard. The right-hand side shows <b>Questions for you to answer</b>: what the AI would like to know that the note doesn't
say. Case T02 (stroke) asks, for example, when the patient was last known to be well. Click a question to answer it in
the box above.</p>
{fig("03_charting_questions", "After Submit: the patient's bed (P9 here), a link to the case file, and the answer box.", 72)}
{fig("03b_charting_stroke_questions", "Case T02 (stroke): the AI's questions. Clicking the first one put it in the answer box, "
     "followed by the answer; Add files it in the case file.", 72)}

<div class="page"></div>
<h3>A3. Read the case file</h3>
{fig("04_case_file", "The case file for T01. Top: level, urgent flag, the levels it can't rule out, NEWS2. Tiles: the numbers. "
     "Left: the tidy case file. Right: questions and to-dos.")}
<ul class="steps">
<li><b>Level box</b> (top left): 1 = most urgent ... 5 = least. <span class="chip red">Urgent: flagged</span> means the
urgency score or the vital-sign safety floor says level 1-2. The orange ⬆ line explains when a level was raised.</li>
<li><b>Tiles</b>: red = dangerous, amber = abnormal, grey = not in the note. Under each number is the exact text it was read
from, so you can check it. Hover a tile for details.</li>
<li><b>Case file</b>: the note filed under standard headings. Press <b>✎ Edit the case file</b> to change anything; your edits are kept
when new notes arrive. <b>Triage decision support</b> at the bottom is always the trained models' result, word for word.</li>
</ul>

<h3>A4. Add information later</h3>
<p>Type in the <b>Add information</b> bar (for example the T01 follow-up: <i>Trop 45 ng/L. ECG ST depression V4-V6.
Aspirin 300mg given, cardiology called</i>) and press <b>Add</b>. The new lines go into the right sections
(Investigations, Plan) and nothing already there is changed. You can add while the AI is still busy.</p>
{fig("05_case_added", "After Add: the trop and ECG are under Investigations, aspirin under Plan and actions.", 80)}

<div class="page"></div>
<h3>A5. Fix a number</h3>
<p>If a number was read wrongly, press <b>Fix numbers</b>, change it, and press <b>Save and re-score</b>. The urgency
and level are scored again, the timeline records who changed what, and the correction is saved for learning.</p>
{fig("06_fix_numbers", "Fix numbers: every field can be edited.", 90)}
{fig("07_after_fix", "After changing systolic BP to 82: re-scored straight away.", 90)}
<p class="muted">Both the fixed rules and MedGemma read every note. If they disagree,
that tile gets a dashed amber border and a <span class="chip amber">check</span> tag, and the app uses the rules' value
until a person presses Fix numbers.</p>

<h3>A6. To-dos, chat and timeline</h3>
<p><b>Questions and to-dos</b> (right): tick a to-do when it is done; ✦ = suggested by the AI. Add your own with
<b>Add a to-do</b>. <b>Chat</b>: ask about the patient; the AI reads the whole case file, including your edits.
The <b>Timeline</b> tab lists every action with time and name, and
<b>Notes as typed</b> keeps the original words.</p>
{fig("08_chat", "Chat: the AI answers from the whole case file.", 80)}
{fig("09_timeline", "Timeline: who did what, and when.", 80)}

<div class="page"></div>
<h2>Part B. Pictures</h2>
<p>Drop pictures on the dashed box before Submit, or use <b>📎 Image</b> in the Add bar later. The app first sorts the
picture, then runs the trained model for that kind. Open the <b>Files</b> tab to see the results. Videos are refused:
upload a still picture.</p>
<table>
<tr><th>Picture</th><th>What runs</th><th>Known accuracy (held-out tests)</th></tr>
<tr><td>Any</td><td>Image sorter: bone X-ray, chest X-ray, CT, MRI, ultrasound, ECG, other</td><td>93.5% right (3,520 test pictures)</td></tr>
<tr><td>Bone X-ray</td><td>Fracture detector. The cut-off depends on age: <b>child</b> (&lt; 18) or <b>adult</b> (or age unknown)</td>
<td>Children's wrists: AUROC 0.99. Adults: catches 82.5%, but marks about 1 in 4 normal X-rays</td></tr>
<tr><td>Chest X-ray</td><td>Chest detector (14 findings with boxes). Only ED findings are red; long-standing ones are listed as "•"</td>
<td>Effusion AUROC 0.97, enlarged heart 0.98; pneumonia (children) caught 85%</td></tr>
<tr><td>Chest X-ray: pneumothorax</td><td><span class="chip red">⚠ red</span> = the detector drew a box;
<span class="chip amber">? Possible pneumothorax</span> = only a second model (CheXpert) thinks so: look at the lung edges</td>
<td>Red alone catches 53% and almost never on a normal X-ray; red or amber catches 87%, but amber shows on about half of all chest X-rays</td></tr>
<tr><td>CT</td><td>Head-CT bleed check, <b>only if the notes say head</b> (CT head, fall, GCS, headache ...). Box = rough area</td>
<td>Per patient AUROC 0.99 on its own test set (0.985 on plain grey slices); only 4 outside pictures checked</td></tr>
<tr><td>MRI, ultrasound, ECG, other</td><td>No trained detector: with the real AI, MedGemma's checklist only</td><td>-</td></tr>
</table>
<p><b>Chips under each picture:</b> <span class="chip red">⚠ red</span> above the model's cut-off: look here ·
<span class="chip amber">? / • amber</span> possible, or seen but usually not urgent · <span class="chip green">✓</span> not found.
The pictures are: <i>As uploaded</i>, <i>AI marks</i> (boxes) and <i>AI outline</i> (MedSAM traces the strongest box).
A mark means <b>look here</b>, not a diagnosis.</p>
{fig("10_pictures_fracture", "Case P01: adult ankle X-ray. The fracture is marked (adult cut-off) and outlined.")}
<div class="page"></div>
{fig("11_pictures_chest", "Case P06: large left pneumothorax. Red pneumothorax with a box and an outline. Effusion and "
     "pneumonia-type shadowing are also red here: false flags happen, which is why a person decides.")}
{fig("12_pictures_head_ct", "Case P11: head CT with an epidural bleed. The bleed check ran because the note says CT head; "
     "the box is a rough area.")}

<div class="page"></div>
<h2>Part C. The whiteboard</h2>
<p>Open <b>Whiteboard</b> (best on a second screen). Patients are sorted most urgent first, and it updates by itself.
The triage nurse or organising doctor can:</p>
<ul class="steps">
<li>change the <b>Stage</b> (Waiting for doctor, Being seen, Results pending, Going to surgery, Transfer, Admitted, Discharged home);</li>
<li>type a <b>Doctor</b>'s name (Enter saves it);</li>
<li>tick a to-do chip, remove one (×), or add one (+ to-do);</li>
<li>when a patient leaves (the last four stages): optionally type the <b>final diagnosis</b> and pick the <b>level that was
right</b>, then press <b>✓ Clear from board</b>. The patient moves to <b>Left the ED</b>, and the case file is kept.</li>
</ul>
{fig("13_board", "The whiteboard during the test (9 patients). The line under the title shows the learning log.")}
<div class="page"></div>
{fig("14_board_outcome", "A patient set to Discharged home: final diagnosis and 'Right level?' boxes, then Clear from board.")}
<div class="box"><b>Learning from use.</b> The outcome boxes, number fixes and AI to-dos are saved in a learning log. The
models never change while the app is running. After at least 20 patients with an outcome, an update can be run offline
(<code>python reasoning/src/learning.py --update</code>). It only replaces a model if it passes a fixed safety test on the
original test patients, it keeps the old version, and <code>--rollback</code> restores it.</div>
{fig("15_board_doctor", "Signed in as a Doctor: the board is view-only.", 90)}

<div class="page"></div>
<h2>Part D. What we would like you to test</h2>
<ol class="steps">
<li><b>Cases</b>: run as many cases from <span class="mono">cases/</span> as you can (P cases need the picture from
<span class="mono">pictures/</span>). Compare with the tables below and <span class="mono">expected_results.csv</span>.</li>
<li><b>Workflow</b>: repeat Part A with any case: add information, answer a question, fix a number, tick to-dos, change
stages, clear a patient with an outcome.</li>
<li><b>Stress it</b>:
<ul>
<li>two or three of you submit patients at the same time;</li>
<li>add a note while the AI is still working;</li>
<li>keep the whiteboard open in two browsers and change it in one;</li>
<li>try very long notes, notes in capitals, typos, impossible numbers, an empty note and a wrong file type;</li>
<li>try 20+ patients in a row and see if the whiteboard stays quick.</li>
</ul></li>
<li><b>Time it</b>: for every case, write in the <span class="mono">time_taken_s</span> column roughly how many seconds it
took from Submit until the case file appeared (a phone stopwatch is fine). Time "Save and re-score", "Add" and one chat
answer too, and say which computer you used. The full AI can take several minutes a case; we need real times.</li>
<li><b>Tell us</b> in <span class="mono">feedback_form.csv</span>: what you did, what happened, whether it matched, and how
bad it was (1 = cosmetic ... 5 = could harm a patient). Screenshots help (name them after the case, e.g. <span class="mono">P06_1.png</span>).
Also: what was confusing, what took too long, and what you would want as a nurse or doctor.</li>
</ol>

<h3>Triage and number reading (T cases)</h3>
{rows_table(rows, "T")}
<p class="muted">T10a, T10b and T10c are the same patient written three ways; they must give the same numbers, level and flag.</p>
<div class="page"></div>
<h3>Pictures (P cases)</h3>
{rows_table(rows, "P")}
<p class="muted">P02 (a fracture the detector misses), P03 (a normal adult X-ray it marks) and P10 (a normal child's chest
X-ray with a false shadowing flag) are left in on purpose: they show what the published accuracy figures mean in
practice.</p>
<h3>Real held-out patients (R cases)</h3>
<p>Patients the models never saw, with the level the real triage nurse gave. In this run the app's level matched
{agree} of the {n_real - 2} with a known level (R01 and R07 are case reports without one). The app's
<i>urgent flag</i> is set to catch about 9 in 10 urgent patients, so it over-flags some.</p>
{rows_table(rows, "R")}

<h2>Part E. Known limits (so you don't need to report these)</h2>
<ul class="steps">
<li>The case-file wording, questions, to-dos and chat are written by MedGemma, so they can be worded a little
differently each time you run the same note. The level, urgent flag, numbers and picture marks should not change.</li>
<li>The urgent flag misses about 1 in 10 urgent patients (KTAS) to 1 in 6 (NHAMCS) in our tests. Catching 19 in 20 would
mean flagging about 70% of all patients, so we kept the current cut-off.</li>
<li>Image models miss findings and flag normal pictures (see the accuracy table). Pictures from journals (with arrows,
labels and crops) are harder than real scans. The head-CT bleed model says "possible bleed" but not which type: its
type guesses were unreliable.</li>
<li>A pathway (chest pain, stroke, sepsis ...) can be suggested that doesn't fit the complaint, with its questions
and to-dos; the strongest one is usually right. Patients under 16 get a children's urgency score (trained on 1,256 children); NEWS2 and the level are still adult
models, and the label says so.</li>
</ul>
<p class="muted">How this guide was checked: every case in the tables was sent to the running app by
<span class="mono">make_test_pack.py</span>, and every screenshot comes from a scripted run of the real screens,
which also checked {sum(c.startswith('PASS') for c in checks)} of {len(checks)} steps
({', '.join(c[6:] for c in checks if c.startswith('FAIL')) or 'all passed'}).</p>
</body></html>"""
    (PACK / "guide.html").write_text(page, encoding="utf-8")
    pdf = PACK / "ED_app_user_guide.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (PACK / "guide.html").as_uri()], check=True, capture_output=True, timeout=120)
    shutil.copy(pdf, OUT / pdf.name)
    zpath = PACK / "ED_app_test_pack.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT.parent))
    print(f"wrote {pdf} ({pdf.stat().st_size / 1e6:.1f} MB) and {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
