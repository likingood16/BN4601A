"""The Gradio app as a Hugging Face Space (9 Nov 2026): code only; the trained files stay in the group's private model
repo (install_models.HUB) and the Space downloads them when it starts, with its HF_TOKEN secret.

    python make_space_bundle.py                      # -> app/build/space/  (the Space's files)
    python make_space_bundle.py --push               # also creates / updates the Space (ZeroGPU) and uploads them
    python make_space_bundle.py --push --space <org>/bn4601a-ed-triage --public   # into the course's HF org

In the Space (same folders as the repo; README.md + requirements.txt at the top, the app is app/gradio/app.py):
    README.md (the Space settings: app_file = app/gradio/app.py), requirements.txt
    reasoning/src, app/server.py + install_models.py + gradio/app.py, vitals/src + models,
    vision/*.py and cut-offs                                            the code, as committed in git
Not in it: MedGemma (the Space downloads the ungated copy unsloth/medgemma-1.5-4b-it), the heads, adapters and image
models (private model repo), data, notebooks, outputs.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "app" / "build" / "space"
SPACE = "likingood/bn4601a-ed-triage"

CODE = ["reasoning/src", "app/server.py", "app/install_models.py", "app/gradio/app.py",
        "vitals/src", "vitals/models", "vision"]
SKIP = ("/notebooks/", "/data/", "/outputs/", "/__pycache__/", ".ipynb", "/test_images/")


def tracked(prefixes):
    """Files as committed in git under these paths (so the Space runs what is on GitHub)."""
    out = subprocess.run(["git", "ls-files", *prefixes], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if not any(s in f"/{f}" for s in SKIP) and not f.endswith(".pt")]


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copy(REPO / "app" / "gradio" / "README_SPACE.md", OUT / "README.md")
    shutil.copy(REPO / "app" / "gradio" / "requirements.txt", OUT / "requirements.txt")
    for f in tracked(CODE):
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, OUT / f)
    files = [f for f in OUT.rglob("*") if f.is_file()]
    print(f"wrote {OUT}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")


def push(space=SPACE, private=True):
    from huggingface_hub import HfApi
    api = HfApi()
    # a Gradio Space on the free CPU needs PRO (Nov 2026); on ZeroGPU it is free
    api.create_repo(space, repo_type="space", space_sdk="gradio", private=private, space_hardware="zero-a10g",
                    exist_ok=True)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(OUT), delete_patterns=["*"],
                      commit_message="ED triage app (Gradio), code from git")
    try:
        api.request_space_hardware(space, "zero-a10g")             # ZeroGPU (its hardware name on the hub)
    except Exception as e:
        print("couldn't switch to ZeroGPU:", e)
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true", help="with --push: make the Space public")
    ap.add_argument("--space", default=SPACE, help="with --push: <owner or org>/<space name>, e.g. the course org")
    a = ap.parse_args()
    build()
    if a.push:
        push(a.space, private=not a.public)
