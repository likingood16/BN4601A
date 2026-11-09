"""User guide for the Gradio version of the app (9 Nov 2026) -> test_pack/ED_app_gradio_guide.pdf

    python make_gradio_guide.py

The screenshots (test_pack/guide_shots/16-23_gradio_*.png) were taken while following these steps by hand in Chrome
on the test-mode Gradio app (reasoning/gradio_app/app.py --test), so the guide shows what the app really does.
"""
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE / "test_pack"
SHOTS = PACK / "guide_shots"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def fig(name, caption, width=100):
    return f'<figure style="width:{width}%"><img src="guide_shots/{name}.png"><figcaption>{caption}</figcaption></figure>'


PAGE = f"""<!doctype html><html><head><meta charset="utf-8"><title>ED triage app: Gradio version guide</title>
<style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; color: #1d2733; line-height: 1.45; }}
h1 {{ font-size: 22pt; margin: 0 0 4px; color: #0d47a1; }}
h2 {{ font-size: 14pt; color: #0d47a1; border-bottom: 2px solid #bbdefb; padding-bottom: 3px; margin-top: 20px;
      page-break-after: avoid; }}
h3 {{ font-size: 12pt; margin: 12px 0 4px; page-break-after: avoid; }}
figure {{ margin: 8px 0 12px; page-break-inside: avoid; }}
figure img {{ width: 100%; border: 1px solid #cfd8dc; border-radius: 6px; }}
figcaption {{ font-size: 9pt; color: #546e7a; margin-top: 3px; }}
.box {{ background: #e3f2fd; border-left: 4px solid #1565c0; padding: 8px 12px; margin: 10px 0; page-break-inside: avoid; }}
.warn {{ background: #fff3e0; border-left: 4px solid #ef6c00; padding: 8px 12px; margin: 10px 0; }}
code {{ font-family: Consolas, monospace; background: #f1f4f7; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }}
pre {{ font-family: Consolas, monospace; background: #f1f4f7; padding: 8px 10px; border-radius: 4px; font-size: 9.5pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 6px 0 10px; }}
th, td {{ border: 1px solid #cfd8dc; padding: 4px 6px; text-align: left; vertical-align: top; }}
th {{ background: #eceff1; }}
li {{ margin-bottom: 3px; }}
.page {{ page-break-before: always; }}
</style></head><body>
<h1>ED triage app: the Gradio version</h1>
<div style="font-size:12pt;color:#455a64">User guide and test plan · BN4601A, Model 3 · version of 9 Nov 2026</div>
<div class="warn"><b>Student research prototype.</b> Not a medical device. Never use it with real patients' details.</div>

<h2>What it is</h2>
<p>The <b>same app</b> as the main version (same back end, same trained models, same answers), with simpler
<b>Gradio</b> screens in one browser tab: <b>Charting</b>, <b>Case file</b> and <b>Whiteboard</b>. We built it because a
Hugging Face Space with a free GPU (ZeroGPU) only takes Gradio apps: on such a Space the full AI, including MedGemma,
can run online for the presentation, with nothing to install. On your laptop it runs in <b>test mode</b>, like the main
app: the number reading, urgency and level scoring, image models and whiteboard are real; MedGemma's writing (case-file
wording, chat answers, picture descriptions) is replaced by text marked [TEST MODE] because it needs a big GPU.</p>

<h2>1. Start it (after installing the main app)</h2>
<p>It uses the same files and the same install as the main app (section 1 of <code>ED_app_user_guide.pdf</code>; the
install already includes Gradio). Then, in Anaconda Prompt / Terminal:</p>
<pre>conda activate edapp
cd Downloads\\EDapp\\ED_app_code\\BN4601A\\reasoning\\gradio_app
python app.py --test</pre>
<p>Wait for <b>Running on local URL: http://127.0.0.1:7860</b>, then open <b>http://127.0.0.1:7860</b> in Chrome or Edge.
The main app (port 8000) can run at the same time. <b>First thing every time:</b> type <b>your name</b> top right and pick
your <b>role</b>; without a name, Submit and whiteboard changes are refused with a warning.</p>

<h2>2. One patient from start to finish (case T01)</h2>
<h3>Charting tab: write the note and press Submit</h3>
<p>Type the note, or use <b>📄 Open a text file</b> and pick <code>cases/T01.txt</code> from the test pack (only .txt files
are accepted). Pictures go in <b>📎 X-rays or photos</b>. To add to a patient who is already in, pick them in the
<b>Patient</b> list (type to search) instead of "New patient".</p>
{fig("16_gradio_charting", "The Charting tab with case T01 typed in.")}
<h3>Case file tab</h3>
<p>Submit takes you here. Top: the <b>level</b>, the <b>urgent flag</b> with its chance of level 1-2, the levels it can't
rule out, NEWS2, stage and doctor. The table shows each number, the exact words it was read from, and who read it.</p>
{fig("17_gradio_case_top", "Case T01: level 2, urgent (the vital-sign safety floor raised it), every number with its source.")}
<p>Below: <b>Fix numbers</b>, <b>Download the whole case file</b> (one click saves a .txt), and <b>Add information</b>
(text and pictures go into the right sections; nothing already there changes). The case file has its own tabs:
<b>Case file</b> (✎ Edit, then Save), <b>Notes as typed</b>, <b>Files</b> (pictures) and <b>Timeline</b> (who did what,
when). On the right: <b>AI questions</b> (click one, it goes into the answer box, type the answer and press Add),
<b>to-dos</b> (tick when done; ✦ = suggested by the AI; add your own; remove one) and <b>Chat</b>.</p>
{fig("18_gradio_case_body", "The case file, filed under standard headings; the to-dos on the right.")}

<div class="page"></div>
<h3>Fix a number</h3>
<p>Press <b>Fix numbers</b>, <b>double-click</b> a value in the table, type the new one, press <b>Enter</b>, then
<b>Save and re-score</b>. The urgency and level are scored again, the tile says "fixed by &lt;your name&gt;", and the
timeline records it.</p>
{fig("19_gradio_fix", "Fix numbers: every field can be changed; the urgency and level are scored again.")}
<h3>Chat</h3>
<p>Ask about the patient in the <b>Chat</b> tab and press <b>Ask</b>. The AI reads the whole case file, including your
edits. (In test mode the answer is a placeholder that repeats the case file.)</p>
{fig("20_gradio_chat", "Chat in test mode.", 80)}

<h2>3. Pictures</h2>
<p>Add a picture when you submit (or later with 📎 Image under Add information). The app sorts it (bone X-ray, chest X-ray,
CT ...) and runs the trained model for that kind. The <b>Files</b> tab shows the picture as uploaded, the <b>AI marks</b>
(boxes) and the <b>AI outline</b>, with the findings under it: ⚠ = above the model's cut-off, look here; ? / • = possible
or seen but usually not urgent; ✓ = not found. A mark means "look here", not a diagnosis. Try case <b>P01</b>
(<code>cases/P01.txt</code> + <code>pictures/adult_ankle_fracture.jpg</code>).</p>
{fig("21_gradio_files", "Case P01: the ankle fracture marked by the fracture detector and outlined.")}

<div class="page"></div>
<h2>4. Whiteboard tab</h2>
<p>Everyone in the ED, most urgent first; it refreshes by itself every few seconds. Only the <b>Triage nurse</b> and the
<b>Organising doctor</b> can change it: pick the patient, set the <b>Stage</b> and <b>Doctor</b>, press <b>Update
board</b>. When a patient leaves (going to surgery, transfer, admitted, discharged), the outcome boxes appear: type the
final diagnosis and pick the level that was right (both optional), then <b>✓ Clear from board</b>. <b>Left the ED</b>
lists them; <b>Put back on the board</b> undoes it. Their case files stay readable in the Case file tab.</p>
{fig("22_gradio_board", "The whiteboard with two patients.")}
{fig("23_gradio_board_leave", "P1 set to Discharged home by Dr Lim: the outcome boxes and Clear from board appear.")}

<h2>5. What we would like you to test (15-20 minutes)</h2>
<ul>
<li>Cases <b>T01</b> (chest pain), <b>T02</b> (stroke: it should ask when she was last known to be well) and <b>P01</b>
(picture). The answers should match <code>expected_results.csv</code> in the test pack, as the back end is the same.</li>
<li>Every button once: Open a text file, Submit, a question → answer → Add, tick and add a to-do, Fix numbers, Edit and
Save, Download, Chat, the whiteboard stage, doctor, Clear and Put back.</li>
<li>Try to break it: Submit without a name, open a picture with "Open a text file", the app open in two browser tabs at
once, a long or messy note.</li>
<li>Write what you found in <code>feedback_form.csv</code> (in the test pack) with <b>Gradio</b> in the "app" column.</li>
</ul>

<h2>6. Differences from the main app, and known limits</h2>
<ul>
<li>One browser tab with three tabs, instead of a charting page, a case-file window and a whiteboard page.</li>
<li>Fix numbers is a table: double-click a value to change it.</li>
<li>In test mode the chat answer is a placeholder, and the questions and to-dos come from fixed lists.</li>
<li>On a Hugging Face ZeroGPU Space (not tried yet), each patient's AI run is one GPU call, and each viewer has a daily
GPU allowance (about 5 minutes free, 40 with PRO): about 5-10 patients a day per free viewer.</li>
</ul>
</body></html>"""


def main():
    from make_guide import trim
    for png in SHOTS.glob("*_gradio_*.png"):
        trim(png)
    html = PACK / "gradio_guide.html"
    html.write_text(PAGE, encoding="utf-8")
    pdf = PACK / "ED_app_gradio_guide.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html.as_uri()], check=True, capture_output=True, timeout=120)
    shutil.copy(pdf, PACK / "ED_app_test_pack" / pdf.name)
    print(f"wrote {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
