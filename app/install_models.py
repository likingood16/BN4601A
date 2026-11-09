"""Model files for the ED app. They are not on GitHub: too big, and some were trained on data that can't be shared
publicly (VinDr-CXR, RSNA). Keep the zip inside the group.

    python install_models.py                            # group member: finds the parts (next to the code, Downloads,
                                                        # Telegram Desktop) and puts each file where the app looks
    python install_models.py ED_app_models.zip          # same, from the whole zip
    python install_models.py path/to/folder             # same, from the split parts (ED_app_models.zip.001, .002 ...)
    python install_models.py --make                     # host: builds build/ED_app_models.zip from this computer
    python install_models.py --make --split 9.5         # host: also cuts it into 9.5 MB parts (Telegram uploads)
    python install_models.py ED_app_full_ai_models.zip  # full AI: the image models + MedGemma's add-ons, one file
    python install_models.py --make --full              # host: builds build/ED_app_full_ai_models.zip

Full AI (MedGemma's add-ons: urgency/level/pathway heads, the case-file, question and number adapters, Cato's image
adapter, plus the image models above) from the group's Hugging Face model repo (access for group members only):
    python install_models.py --hub                      # group member: downloads them all (needs `hf auth login`
                                                        # with an account that was given access)
    python install_models.py --upload                   # host: uploads them from this computer

What's in it: the picture sorter, fracture detector, chest X-ray detector, the two head-CT bleed models and the CheXpert
weights used for the "possible pneumothorax" second opinion. Caleb's vitals model is already in git (vitals/models/).
MedSAM (the outline tool) is not included: without it the app shades the box instead of outlining what is inside.
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()
NAME = "ED_app_models.zip"
FULL = "ED_app_full_ai_models.zip"                              # image models + MedGemma's add-ons (one ~700 MB file)
HUB = "likingood/bn4601a-ed-models"                             # private / gated model repo (group members only)
EMB = "base4bit"                                                # the vectors the heads were trained on (EDPipeline)
sys.path.insert(0, str(REPO / "reasoning" / "src"))
import image_models as im                                       # noqa: E402  (the paths the app reads from)


def targets():
    """Name inside the zip -> where the app looks for the file."""
    out = {}
    for f in (im.SORTER, im.FRACTURE, im.CHEST, im.ICH, im.ICH_GREY, im.CHEX_FILE):
        try:
            out[f.relative_to(REPO).as_posix()] = f
        except ValueError:                                      # the CheXpert file lives in the home folder
            out["home/" + f.relative_to(HOME).as_posix()] = f
    return out


def ai_targets():
    """targets() plus everything the full AI adds on top of MedGemma (name in the hub repo -> where the app looks)."""
    from config import HEADS_DIR, LORA_DIR
    adapters = ["task/stage1", "questions/stage1", "numbers/stage1"]     # pipeline.EXPLAINER, QUESTIONER, NUMBER_READER
    files = [HEADS_DIR / f"urgent__{EMB}__ktas_nhamcs.joblib", HEADS_DIR / f"level__{EMB}.joblib",
             HEADS_DIR / f"pathways__{EMB}.joblib"]
    files += [LORA_DIR / a / "adapter" / n for a in adapters
              for n in ("adapter_config.json", "adapter_model.safetensors", "README.md")]
    cato = REPO / "vision" / "cato" / "hpc_run_3epochs" / "huggingface_adapter"   # image_reader.IMAGE_ADAPTER
    files += [cato / n for n in ("adapter_config.json", "adapter_model.safetensors", "README.md")]
    return {**targets(), **{f.relative_to(REPO).as_posix(): f for f in files}}


def upload(repo=HUB):
    """Host: put every full-AI file in the private model repo (one commit; unchanged files are skipped by the hub)."""
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    files = ai_targets()
    missing = [n for n, f in files.items() if not f.exists()]
    if missing:
        sys.exit("missing on this computer: " + ", ".join(missing))
    print(f"uploading {len(files)} files, {sum(f.stat().st_size for f in files.values()) / 1e6:.0f} MB -> {repo}")
    api.create_commit(repo, operations=[CommitOperationAdd(n, str(f)) for n, f in files.items()],
                      commit_message="ED app model files (heads, adapters, image models)")
    print("done")


def from_hub(repo=HUB, token=None):
    """Download the full-AI files and put each where the app looks. Returns how many files were put in place."""
    import shutil
    from huggingface_hub import snapshot_download
    files = ai_targets()
    try:
        local = Path(snapshot_download(repo, allow_patterns=list(files), token=token))
    except Exception as e:                                      # no login, no access yet, or no internet
        sys.exit(f"Couldn't download from https://huggingface.co/{repo} ({type(e).__name__}).\n"
                 "  1. Make a Hugging Face account and ask Alexander to give it access (or press 'Request access'\n"
                 "     on that page while logged in).\n"
                 "  2. Log in on this computer:  hf auth login   (paste a 'Read' token from huggingface.co/settings/tokens)\n"
                 "  3. Run this again.")
    done = 0
    for name, dest in files.items():
        src = local / name
        if not src.exists():
            print(f"MISSING in {repo}: {name}")
            continue
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            print(f"{name} -> {dest}")
        done += 1
    print(f"{done} of {len(files)} files in place")
    return done


def make(zip_path, split_mb=None, full=False):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = ai_targets() if full else targets()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED if full else zipfile.ZIP_DEFLATED) as z:
        for name, f in files.items():
            z.write(f, name)
            print(f"added {name} ({f.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    if split_mb:
        for old in zip_path.parent.glob(zip_path.name + ".*"):
            old.unlink()
        data, size = zip_path.read_bytes(), int(split_mb * 1e6)
        for i in range(0, len(data), size):
            part = zip_path.with_name(f"{zip_path.name}.{i // size + 1:03d}")
            part.write_bytes(data[i:i + size])
        print(f"cut into {-(-len(data) // size)} parts of up to {split_mb} MB: {zip_path.name}.001 ...")


def _open(src):
    """A whole zip, or the parts .001, .002 ... (given the folder or any one part) joined in order."""
    src = Path(src)
    if src.is_file() and src.suffix == ".zip":
        return zipfile.ZipFile(src)
    folder = src if src.is_dir() else src.parent
    whole = folder / NAME
    parts = sorted(folder.glob(NAME + ".[0-9][0-9][0-9]"))
    if not parts:
        if whole.exists():
            return zipfile.ZipFile(whole)
        sys.exit(f"No {NAME} or {NAME}.001 ... found in {folder}")
    numbers = [int(p.suffix[1:]) for p in parts]
    if numbers != list(range(1, len(parts) + 1)):
        sys.exit(f"Some parts are missing: found {', '.join(p.name for p in parts)}")
    print(f"joining {len(parts)} parts")
    try:
        return zipfile.ZipFile(io.BytesIO(b"".join(p.read_bytes() for p in parts)))
    except zipfile.BadZipFile:
        sys.exit("The joined parts are not a whole zip: the last part may be missing or half-downloaded")


def find_parts():
    """No path given: look where people usually leave the downloads (next to the unzipped code, Downloads, Telegram)."""
    places = [REPO.parent, REPO.parent.parent, Path.cwd(), HOME / "Downloads", HOME / "Downloads" / "Telegram Desktop",
              HOME / "Desktop"]
    for place in places:
        if (place / FULL).exists():
            print(f"found {FULL} in {place}")
            return place / FULL
    for place in places:
        if (place / (NAME + ".001")).exists() or (place / NAME).exists():
            print(f"found the model files in {place}")
            return place
    sys.exit(f"Couldn't find {NAME}.001 ... Put all the parts in one folder and run:\n"
             f"    python install_models.py <that folder>")


def install(src):
    image = targets()
    with _open(src) as z:
        inside = set(z.namelist())
        for name, dest in ai_targets().items():                 # only these names are written, nothing else in the zip
            if name not in inside:
                if name in image:                               # (the full-AI files are only in the full-AI zip)
                    print(f"MISSING from the zip: {name}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            print(f"{name} -> {dest.relative_to(REPO) if dest.is_relative_to(REPO) else dest}")
    ok = {**im.available(), "chexpert": im.CHEX_FILE.exists()}
    full = all(f.exists() for f in ai_targets().values())
    print("image models found:", ok, "| full-AI files:", "all there" if full else "not installed")
    if not all(ok.values()):
        print("some models are missing (see above)")
    elif full:
        print("all there. Full AI (NVIDIA GPU):  python server.py      Test mode (any computer):  python server.py --test")
    else:
        print("all there for test mode, now run:  python server.py --test   (full AI: python install_models.py " + FULL + ")")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="ED_app_models.zip, or the folder with its parts .001, .002 ...")
    ap.add_argument("--make", action="store_true", help="build the zip from this computer's model files")
    ap.add_argument("--split", type=float, help="with --make: also cut the zip into parts of this many MB")
    ap.add_argument("--full", action="store_true", help="with --make: the full-AI zip (" + FULL + ")")
    ap.add_argument("--hub", action="store_true", help="full AI: download every model file from the group's HF repo")
    ap.add_argument("--upload", action="store_true", help="host: upload every model file to the group's HF repo")
    a = ap.parse_args()
    if a.upload:
        upload(a.src or HUB)
    elif a.hub:
        if from_hub(a.src or HUB) == len(ai_targets()):
            print("all there. Full AI (NVIDIA GPU):  python server.py      Test mode (any computer):  python server.py --test")
    elif a.make:
        make(Path(a.src) if a.src else Path(__file__).resolve().parent / "build" / (FULL if a.full else NAME), a.split,
             a.full)
    else:
        install(a.src or find_parts())
