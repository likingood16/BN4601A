"""Model files for the ED app. They are not on GitHub: too big, and some were trained on data that can't be shared
publicly (VinDr-CXR, RSNA). Keep the zip inside the group.

    python install_models.py ED_app_models.zip      # group member: puts each file where the app looks for it
    python install_models.py --make                 # host: builds test_pack/ED_app_models.zip from this computer

What's in it: the picture sorter, fracture detector, chest X-ray detector, the two head-CT bleed models and the CheXpert
weights used for the "possible pneumothorax" second opinion. Caleb's vitals model is already in git (vitals/models/).
MedSAM (the outline tool) is not included: without it the app shades the box instead of outlining what is inside.
"""
import argparse
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
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


def make(zip_path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, f in targets().items():
            z.write(f, name)
            print(f"added {name} ({f.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


def install(zip_path):
    with zipfile.ZipFile(zip_path) as z:
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
    ap.add_argument("zip", nargs="?", help="ED_app_models.zip from the group chat")
    ap.add_argument("--make", action="store_true", help="build the zip from this computer's model files")
    a = ap.parse_args()
    if a.make:
        make(Path(a.zip) if a.zip else Path(__file__).resolve().parent / "test_pack" / "ED_app_models.zip")
    elif a.zip:
        install(Path(a.zip))
    else:
        ap.print_help()
