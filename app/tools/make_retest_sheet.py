"""Two-page "how to retest" sheet for the group (13 Nov 2026) -> app/build/ED_app_retest_steps.pdf

    python make_retest_sheet.py

Goes with the Telegram message: what to download, how to update, what to run, which case shows each fix, and the
timing request. Same steps as the message, so the two never disagree.
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "build"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

FIXES = [   # (what was reported on 10 Nov, what you should see now, case to check)
    ("Off-topic pathways (asthma for chest pain; poisoning, collapse for a head injury)",
     "Only the pathway the note supports: chest pain for T01; no poisoning or collapse to-do for P11 or T05", "T01, P11, T05"),
    ("Urgent with no reason when observations are missing",
     'Amber "Needs vitals: BP and RR not recorded" (it says what to measure); the first question asks for the '
     'observations', "T09, P05"),
    ("Children scored with adult models",
     "A children's urgency score (new); NEWS2 and level still adult, and the label says so. P10, a well 5-year-old, "
     "is now level 4 (was urgent)", "P04, P05, P10"),
    ("Small finger cut got primary survey + X-ray to-dos",
     'No primary survey or X-ray to-do: just "Check tetanus status" and "Repeat full set of observations"', "T04"),
    ("Ambulance numbers looked like arrival numbers", 'Case file says "Before arrival (ambulance crew): ..."', "T05"),
    ("X-ray to-do although a CT was attached", "No X-ray to-do when a CT is attached", "P11, P12"),
    ("Labels unclear", '"Questions for you to answer", "Add my answer", "✎ Edit the case file"', "any"),
    ("Fix numbers left the old number in the case file",
     'Observations gets "Corrected by <your name>: BP sys 82 (was 90)"', "T01 + Fix numbers"),
    ("Chat gave made-up reasons", "The chat uses the case's real numbers and reasons", "T01, P05 + Chat"),
    ("Gradio whiteboard confusing",
     '"Change this patient" box: pick patient, stage, doctor, then "Save the change"; "Show patients who left the ED"',
     "Gradio, Whiteboard tab"),
]

ROWS = "\n".join(f"<tr><td>{a}</td><td>{b}</td><td class='c'>{c}</td></tr>" for a, b, c in FIXES)

PAGE = f"""<!doctype html><html><head><meta charset="utf-8"><title>ED app: how to retest</title>
<style>
@page {{ size: A4; margin: 14mm 14mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; color: #1d2733; line-height: 1.42; }}
h1 {{ font-size: 19pt; margin: 0 0 2px; color: #0d47a1; }}
h2 {{ font-size: 12.5pt; color: #0d47a1; border-bottom: 2px solid #bbdefb; padding-bottom: 2px; margin: 14px 0 6px; }}
.sub {{ color: #555; margin: 0 0 8px; }}
ol li {{ margin-bottom: 5px; }}
code, .mono {{ font-family: Consolas, monospace; background: #f1f4f8; padding: 0 3px; border-radius: 3px; font-size: 9.3pt; }}
pre {{ font-family: Consolas, monospace; background: #f1f4f8; padding: 6px 9px; border-radius: 4px; margin: 4px 0; font-size: 9.3pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; }}
th, td {{ border: 1px solid #cfd8dc; padding: 4px 6px; vertical-align: top; text-align: left; }}
th {{ background: #e3f2fd; }}
td.c {{ white-space: nowrap; font-weight: 600; }}
.box {{ border: 2px solid #ffb300; background: #fff8e1; border-radius: 6px; padding: 7px 10px; margin: 8px 0; }}
.page {{ page-break-before: always; }}
</style></head><body>
<h1>ED triage app: how to retest (13 Nov)</h1>
<p class="sub">Your 10 Nov feedback is fixed, and two urgency models were retrained. Full AI only. About 1-2 hours.
Please test it and send the form back <b>by tonight (Fri 13 Nov)</b>, so there is the weekend to fix things before the 16 Nov deadline. Even a few cases help: start with T01-T10.</p>

<h2>1. Update the app (10 minutes)</h2>
<ol>
<li>Download from the group chat: <code>ED_app_code.zip</code>, <code>ED_app_models_update.zip</code>,
<code>ED_app_test_pack.zip</code>, and this sheet and the two guides if you want them.</li>
<li>Unzip <code>ED_app_code.zip</code> and use it instead of your old app folder.</li>
<li>Models: keep <code>ED_app_full_ai_models.zip</code> from 10 Nov (no need to download it again; if you no longer
have it, it is in the chat from 10 Nov). Copy both model zips into the new <code>BN4601A/app</code> folder, open a
terminal there and run:
<pre>python install_models.py ED_app_full_ai_models.zip
python install_models.py ED_app_models_update.zip</pre>
The second one adds the retrained urgency score and the new children's urgency score (0.1 MB).</li>
<li>Start it, in the same folder: <code>python server.py</code> (main app, then open the address it prints) or
<code>python gradio/app.py</code> (Gradio). The top of the page must say <b>Full AI</b>. Install details and
problems: section 1 of <code>ED_app_user_guide.pdf</code>.</li>
</ol>

<h2>2. Run the cases (1-2 hours)</h2>
<ol>
<li>Unzip <code>ED_app_test_pack.zip</code> and read <code>README.txt</code>.</li>
<li>Run <b>T01-T10</b> first, then <b>P01-P18</b> (note + the picture named in <code>expected_results.csv</code>), then
<b>R01-R10</b>. Open a note with "Open a text file" or copy-paste it.</li>
<li>Compare each with <code>expected_results.csv</code>: the <b>level</b>, the <b>urgent flag</b>, the <b>numbers</b>
and the <b>picture marks</b> should match. The case-file wording, questions, to-dos and chat are written by MedGemma
and can be worded a little differently each time: judge whether they are right and useful.</li>
<li>Try the workflow once: add information, answer a question, Fix numbers, Chat, then on the whiteboard set a
stage and a doctor and clear a patient with an outcome.</li>
</ol>

<div class="box"><b>New this time: please time it.</b> For each case write roughly how many seconds it took from
<b>Submit</b> until the <b>case file appeared</b> in the <code>time_taken_s</code> column (a phone stopwatch is
fine). Also time one <b>Save and re-score</b>, one <b>Add</b> and one <b>chat answer</b>, and write in comments which
computer you used (laptop + GPU, Mac, or Hugging Face). The full AI can take several minutes a case; we need real
times for the report.</div>

<h2>3. Send it back</h2>
<p>Fill in <code>feedback_form.csv</code> (one row per case or step: what you did, what happened, same as the guide
yes/no, how bad 1 cosmetic - 5 unsafe, time) and send it in the group with screenshots named by case
(<code>P06_1.png</code>).</p>

<div class="page"></div>
<h2>4. Please re-check what you reported on 10 Nov</h2>
<table><tr><th>You reported</th><th>What you should see now</th><th>Check with</th></tr>
{ROWS}
</table>

<h2>5. Still known (not fixed yet)</h2>
<ul>
<li>Some pictures get a wrong mark: P03 (fracture mark on a normal X-ray), P07 and P10 (possible pneumothorax).
These are the image models; a mark means "look here", not a diagnosis.</li>
<li>Some real level-3 patients are flagged urgent (R03-R06): the cut-off is set to catch 9 in 10 urgent patients, so
there are more false alarms.</li>
<li>Some non-stroke cases still get a stroke to-do (T03, T07, P11): left on purpose, missing a stroke is worse. A
retrained pathway model caught more but also flagged more, so it was not used.</li>
</ul>
<p class="sub">Student research prototype, not a medical device: never use real patients' details.</p>
</body></html>"""


def main():
    html = PACK / "retest_steps.html"
    html.write_text(PAGE, encoding="utf-8")
    pdf = PACK / "ED_app_retest_steps.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html.as_uri()], check=True, capture_output=True, timeout=120)
    print(f"wrote {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
