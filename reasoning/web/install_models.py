"""Model files for the ED app. They are not on GitHub: too big, and some were trained on data that can't be shared
publicly (VinDr-CXR, RSNA). Keep the zip inside the group.

    python install_models.py                            # group member: finds the parts (next to the code, Downloads,
                                                        # Telegram Desktop) and puts each file where the app looks
    python install_models.py ED_app_models.zip          # same, from the whole zip
    python install_models.py path/to/folder             # same, from the split parts (ED_app_models.zip.001, .002 ...)
    python install_models.py --make                     # host: builds test_pack/ED_app_models.zip from this computer
    python install_models.py --make --split 9.5         # host: also cuts it into 9.5 MB parts (Telegram uploads)

What's in it: the picture sorter, fracture detector, chest X-ray detector, the two head-CT bleed models and the CheXpert
weights used for the "possible pneumothorax" second opinion. Caleb's vitals model is already in git (vitals/models/).
MedSAM (the outline tool) is not included: without it the app shades the box instead of outlining what is inside.
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
NAME = "ED_app_models.zip"
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


def make(zip_path, split_mb=None):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, f in targets().items():
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
        if (place / (NAME + ".001")).exists() or (place / NAME).exists():
            print(f"found the model files in {place}")
            return place
    sys.exit(f"Couldn't find {NAME}.001 ... Put all the parts in one folder and run:\n"
             f"    python install_models.py <that folder>")


def install(src):
    with _open(src) as z:
        inside = set(z.namelist())
        for name, dest in targets().items():                    # only these names are written, nothing else in the zip
            if name not in inside:
                print(f"MISSING from the zip: {name}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            print(f"{name} -> {dest}")
    ok = {**im.available(), "chexpert": im.CHEX_FILE.exists()}
    print("models found:", ok)
    print("all there, now run:  python server.py --test" if all(ok.values()) else "some models are missing (see above)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="ED_app_models.zip, or the folder with its parts .001, .002 ...")
    ap.add_argument("--make", action="store_true", help="build the zip from this computer's model files")
    ap.add_argument("--split", type=float, help="with --make: also cut the zip into parts of this many MB")
    a = ap.parse_args()
    if a.make:
        make(Path(a.src) if a.src else Path(__file__).resolve().parent / "test_pack" / NAME, a.split)
    else:
        install(a.src or find_parts())
