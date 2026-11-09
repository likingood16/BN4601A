"""Everything a group member needs to run the app, cut into files small enough to send in a Telegram chat (< 10 MB),
in one folder per message (9 Nov 2026).

    python make_install_pack.py      # -> app/build/telegram/  (run make_gradio_guide.py and make_guide.py first)

    1_main_app/   message 1: the app on your own computer
        ED_app_user_guide.pdf             how to install it (section 1) and use it, every screen
        ED_app_code.zip                   the code: git archive of the current branch (no data, no model files)
        ED_app_models.zip.001 ... .NNN    the image models, cut into parts (install_models.py joins them again)
        ED_app_test_pack.zip              test cases, pictures, answer key, feedback form (+ both guides)
    2_gradio/     message 2: the Gradio version (same install as message 1)
        ED_app_gradio_guide.pdf
    3_full_ai/    the full AI on your own NVIDIA GPU (one 715 MB file: drag it into Telegram yourself)
        ED_app_full_ai_models.zip
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACK = HERE.parent / "build"
OUT = PACK / "telegram"
PART_MB = 9.5


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    main_app, gradio, full = OUT / "1_main_app", OUT / "2_gradio", OUT / "3_full_ai"
    main_app.mkdir(parents=True)
    gradio.mkdir()
    full.mkdir()
    # the code: what is committed on this branch (so it matches GitHub), as BN4601A/...
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    # the tracked files as they are now (uncommitted edits included, nothing committed): git stash create makes a
    # snapshot commit without touching the branch; with no edits it prints nothing and HEAD is used
    ref = subprocess.run(["git", "stash", "create"], cwd=REPO, capture_output=True, text=True).stdout.strip() or "HEAD"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=BN4601A/", "-o", str(main_app / "ED_app_code.zip"),
                    ref], cwd=REPO, check=True)
    # the image models, cut into parts
    subprocess.run([sys.executable, str(HERE.parent / "install_models.py"), "--make", str(main_app / "ED_app_models.zip"),
                    "--split", str(PART_MB)], check=True)
    (main_app / "ED_app_models.zip").unlink()
    # the guides and the test pack (make_guide.py, make_gradio_guide.py)
    for name in ("ED_app_user_guide.pdf", "ED_app_test_pack.zip"):
        shutil.copy(PACK / name, main_app / name)
    shutil.copy(PACK / "ED_app_gradio_guide.pdf", gradio / "ED_app_gradio_guide.pdf")
    subprocess.run([sys.executable, str(HERE.parent / "install_models.py"), "--make", "--full",
                    str(full / "ED_app_full_ai_models.zip")], check=True, capture_output=True)
    print(f"branch {branch} -> {OUT}")
    for folder in (main_app, gradio, full):
        files = sorted(folder.iterdir())
        print(f"  {folder.name}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")
        for f in files:
            size = f.stat().st_size / 1e6
            print(f"    {f.name:32s} {size:5.1f} MB" + ("   TOO BIG for one upload" if size >= 10 else ""))


if __name__ == "__main__":
    main()
