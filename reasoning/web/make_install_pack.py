"""Everything a group member needs to run the app, cut into files small enough to send in a Telegram chat (< 10 MB).

    python make_install_pack.py          # -> test_pack/telegram/  (run make_guide.py first for the user guide + test pack)

What it makes:
    ED_app_install_guide.pdf          how to install from these files (this script writes it)
    ED_app_code.zip                   the code: git archive of the current branch (no data, no model files)
    ED_app_models.zip.001 ... .NNN    the image models, cut into parts (install_models.py joins them again)
    ED_app_user_guide.pdf             how to use the app (from make_guide.py)
    ED_app_test_pack.zip              test cases, pictures, answer key, feedback form (from make_guide.py)
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACK = HERE / "test_pack"
OUT = PACK / "telegram"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PART_MB = 9.5

GUIDE = """<!doctype html><html><head><meta charset="utf-8"><title>ED triage app: install guide</title>
<style>
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; color: #1d2733; line-height: 1.45; }
h1 { font-size: 22pt; margin: 0 0 4px; color: #0d47a1; }
h2 { font-size: 14pt; color: #0d47a1; border-bottom: 2px solid #bbdefb; padding-bottom: 3px; margin-top: 20px; }
.box { background: #e3f2fd; border-left: 4px solid #1565c0; padding: 8px 12px; margin: 10px 0; }
.warn { background: #fff3e0; border-left: 4px solid #ef6c00; padding: 8px 12px; margin: 10px 0; }
code { font-family: Consolas, monospace; background: #f1f4f7; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { font-family: Consolas, monospace; background: #f1f4f7; padding: 8px 10px; border-radius: 4px; font-size: 9.5pt;
      white-space: pre; overflow: hidden; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 6px 0 10px; }
th, td { border: 1px solid #cfd8dc; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eceff1; }
li { margin-bottom: 4px; }
.muted { color: #607d8b; }
</style></head><body>
<h1>ED triage app: install guide</h1>
<div style="font-size:12pt;color:#455a64">BN4601A group test · install from the files in the Telegram chat · about 15 minutes</div>
<div class="warn"><b>Student research prototype.</b> Not a medical device. Never use it with real patients' details.
Please keep the model files inside the group: some models were trained on data that can't be shared publicly.</div>

<h2>1. Download the files</h2>
<p>Save <b>all</b> of these from the chat into <b>one folder</b>, for example <code>Downloads\\EDapp</code>
(Telegram Desktop saves to <code>Downloads\\Telegram Desktop</code> by default, which also works).</p>
<table><tr><th>File</th><th>What it is</th></tr>
FILE_ROWS
</table>

<h2>2. Install Python 3.11 (skip if you have Anaconda or Miniconda)</h2>
<p>Install <b>Miniconda</b> from <code>docs.conda.io/en/latest/miniconda.html</code> (Windows or Mac, default options).
Then open <b>Anaconda Prompt</b> (Windows: Start menu) or <b>Terminal</b> (Mac).</p>

<h2>3. Unzip the code</h2>
<p>Right-click <code>ED_app_code.zip</code> → <b>Extract All</b> (Mac: double-click). You get a folder <code>BN4601A</code>.</p>

<h2>4. Set up and start (copy these lines one by one)</h2>
<p>Replace the first line's path with where your <code>BN4601A</code> folder is. Tip: type <code>cd </code> and drag the
<code>web</code> folder into the window.</p>
<pre>cd Downloads\\EDapp\\ED_app_code\\BN4601A\\reasoning\\web
conda create -n edapp python=3.11 -y
conda activate edapp
pip install -r requirements_app.txt
python install_models.py
python server.py --test</pre>
<ul>
<li><code>pip install</code> takes 5–10 minutes and about 2 GB (PyTorch and friends). Only the first time.</li>
<li><code>install_models.py</code> finds the <code>ED_app_models.zip.001 ...</code> parts by itself (next to the code, in
Downloads or Telegram Desktop), joins them and puts each model where the app looks. It should end with
<b>"all there"</b>. If it can't find them: <code>python install_models.py "path\\to\\the\\folder"</code>.</li>
<li><code>server.py --test</code> prints <b>Uvicorn running on http://127.0.0.1:8000</b>. Leave the window open.</li>
</ul>
<p>Now open <b>http://127.0.0.1:8000</b> in Chrome or Edge, type <b>your name</b> top right, pick your role, and follow
<b>ED_app_user_guide.pdf</b> (Part A first). The whiteboard is at <code>http://127.0.0.1:8000/#/board</code>.</p>

<h2>Next time</h2>
<pre>conda activate edapp
cd Downloads\\EDapp\\ED_app_code\\BN4601A\\reasoning\\web
python server.py --test</pre>

<h2>If something goes wrong</h2>
<table><tr><th>You see</th><th>Do this</th></tr>
<tr><td>"Some parts are missing: ..."</td><td>Download the missing part(s) from the chat into the same folder.</td></tr>
<tr><td>"The joined parts are not a whole zip"</td><td>A part is half-downloaded: download the last part again.</td></tr>
<tr><td>"error while attempting to bind on address ... 8000"</td><td>Something already uses port 8000:
<code>python server.py --test --port 8001</code>, then open <code>http://127.0.0.1:8001</code>.</td></tr>
<tr><td>pip fails on <code>scikit-learn==1.9.1</code></td><td>Check <code>python --version</code> says 3.11
(run <code>conda activate edapp</code> first).</td></tr>
<tr><td>"python is not recognised"</td><td>Use Anaconda Prompt, not the normal Command Prompt.</td></tr>
<tr><td>Anything else</td><td>Screenshot the window and send it in the chat.</td></tr>
</table>

<h2>Why test mode?</h2>
<p class="muted">The full AI (MedGemma) needs a big graphics card: on a 4 GB laptop GPU one patient takes about 6 minutes.
In test mode everything runs on a normal laptop: the number reading, urgency score, triage level, pathway rules, image
models, whiteboard and learning log are real; only MedGemma's writing (case-file wording, chat answers, picture
descriptions) is replaced by text marked [TEST MODE].</p>
</body></html>"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    # 1. code: what is committed on this branch (so it matches GitHub), as BN4601A/...
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--", "reasoning", "vitals", "vision"], cwd=REPO,
                      capture_output=True, text=True).stdout.strip():
        print("note: uncommitted changes are NOT in the code zip (it is made from the last commit)")
    subprocess.run(["git", "archive", "--format=zip", "--prefix=BN4601A/", "-o", str(OUT / "ED_app_code.zip"), "HEAD"],
                   cwd=REPO, check=True)
    # 2. models, cut into parts
    subprocess.run([sys.executable, str(HERE / "install_models.py"), "--make", str(OUT / "ED_app_models.zip"),
                    "--split", str(PART_MB)], check=True)
    (OUT / "ED_app_models.zip").unlink()
    # 3. user guide + test pack from make_guide.py
    for name in ("ED_app_user_guide.pdf", "ED_app_test_pack.zip"):
        shutil.copy(PACK / name, OUT / name)
    # 4. the install guide, listing the files that are really there
    what = {"ED_app_code.zip": "the app's code", "ED_app_user_guide.pdf": "how to use the app (read after installing)",
            "ED_app_test_pack.zip": "test cases, pictures, answer key and feedback form"}
    parts = sorted(OUT.glob("ED_app_models.zip.*"))
    rows = [f"<tr><td><code>ED_app_install_guide.pdf</code></td><td>this guide</td></tr>"]
    rows += [f"<tr><td><code>{n}</code></td><td>{w} ({(OUT / n).stat().st_size / 1e6:.1f} MB)</td></tr>" for n, w in what.items()]
    rows.append(f"<tr><td><code>{parts[0].name}</code> ... <code>{parts[-1].name}</code></td><td>the image models in "
                f"{len(parts)} parts (all {len(parts)} needed, {sum(p.stat().st_size for p in parts) / 1e6:.0f} MB)</td></tr>")
    html = OUT / "install_guide.html"
    html.write_text(GUIDE.replace("FILE_ROWS", "\n".join(rows)), encoding="utf-8")
    pdf = OUT / "ED_app_install_guide.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html.as_uri()], check=True, capture_output=True, timeout=120)
    html.unlink()
    print(f"branch {branch} -> {OUT}")
    for f in sorted(OUT.iterdir()):
        size = f.stat().st_size / 1e6
        print(f"  {f.name:32s} {size:5.1f} MB" + ("   TOO BIG for one upload" if size >= 10 else ""))


if __name__ == "__main__":
    main()
