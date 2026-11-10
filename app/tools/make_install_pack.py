"""Everything a group member needs to run the app, cut into files small enough to send in a Telegram chat (< 10 MB),
in one folder per message (9 Nov 2026).

    python make_install_pack.py      # -> app/build/telegram/  (run make_gradio_guide.py and make_guide.py first)

    send_to_group/
        ED_app_code.zip                   the code (both apps): the tracked files as they are now
        ED_app_full_ai_models.zip         every trained model (715 MB; drag it into Telegram yourself)
        ED_app_user_guide.pdf             install (section 1) and every screen of the main app
        ED_app_gradio_guide.pdf           the Gradio app
        ED_app_test_pack.zip              test cases, pictures, answer key, feedback form
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
    """The 5 files for the group, in one folder: telegram/send_to_group/."""
    out = OUT / "send_to_group"
    if OUT.exists():
        shutil.rmtree(OUT)
    out.mkdir(parents=True)
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.strip()
    # the tracked files as they are now (uncommitted edits included, nothing committed): git stash create makes a
    # snapshot commit without touching the branch; with no edits it prints nothing and HEAD is used
    ref = subprocess.run(["git", "stash", "create"], cwd=REPO, capture_output=True, text=True).stdout.strip() or "HEAD"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=BN4601A/", "-o", str(out / "ED_app_code.zip"), ref],
                   cwd=REPO, check=True)
    subprocess.run([sys.executable, str(HERE.parent / "install_models.py"), "--make", "--full",
                    str(out / "ED_app_full_ai_models.zip")], check=True, capture_output=True)
    for name in ("ED_app_user_guide.pdf", "ED_app_gradio_guide.pdf", "ED_app_test_pack.zip"):   # make_guide.py etc.
        shutil.copy(PACK / name, out / name)
    print(f"branch {branch} -> {out}")
    for f in sorted(out.iterdir()):
        print(f"    {f.name:32s} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
