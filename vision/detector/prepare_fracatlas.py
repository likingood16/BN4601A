"""FracAtlas -> YOLO segmentation dataset (boxes + outlines of fractures on X-rays).

    python prepare_fracatlas.py            # reads data/raw/FracAtlas, writes data/yolo + data/yolo/fracatlas.yaml

FracAtlas (Abedeen et al., Scientific Data 2023, CC BY 4.0): 4,083 musculoskeletal X-rays (hand, leg, hip, shoulder);
719 fractured with radiologist outlines (COCO polygons) for 924 fractures.
Split:
  - fractured images: the OFFICIAL split (Utilities/Fracture Split: 574 train / 82 valid / 63 test), so results
    compare with the paper;
  - non-fractured images are not in that split. We add them (seed 42) so the detector also learns NOT to mark normal
    X-rays: 574 to train, 82 to valid (balanced), 400 to test (to measure false alarms). The rest is unused.
No patient IDs exist in FracAtlas, so a patient could appear in two splits (stated as a limitation).
"""
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw" / "FracAtlas"
OUT = HERE / "data" / "yolo"
SEED = 42
NEG = {"train": 574, "val": 82, "test": 400}


def polygons_by_image():
    """COCO outlines -> {file_name: ["0 x1 y1 x2 y2 ...", ...]} with coordinates scaled to 0-1."""
    coco = json.loads((RAW / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json").read_text())
    images = {im["id"]: im for im in coco["images"]}
    lines = {}
    for a in coco["annotations"]:
        im = images[a["image_id"]]
        seg = a["segmentation"]
        seg = json.loads(seg) if isinstance(seg, str) else seg
        for poly in seg:                                   # one fracture can have several polygons
            xs, ys = poly[0::2], poly[1::2]
            pts = " ".join(f"{min(max(x / im['width'], 0), 1):.6f} {min(max(y / im['height'], 0), 1):.6f}"
                           for x, y in zip(xs, ys))
            lines.setdefault(im["file_name"], []).append(f"0 {pts}")
    return lines


def opens(path):
    """True if the image file is complete (FracAtlas has at least one cut-off JPEG)."""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (OSError, SyntaxError):
        return False


def main():
    lines = polygons_by_image()
    info = pd.read_csv(RAW / "dataset.csv")
    split_dir = RAW / "Utilities" / "Fracture Split"
    pos = {s: pd.read_csv(split_dir / f"{f}.csv")["image_id"].tolist()
           for s, f in (("train", "train"), ("val", "valid"), ("test", "test"))}
    negatives = info.loc[info["fractured"] == 0, "image_id"].sample(frac=1, random_state=SEED).tolist()
    damaged = [n for n in negatives if not opens(RAW / "images" / "Non_fractured" / n)]
    negatives = [n for n in negatives if n not in damaged]          # skip damaged normal X-rays (seed order kept)
    bad_pos = [n for s in pos for n in pos[s] if not opens(RAW / "images" / "Fractured" / n)]
    print("damaged images skipped:", damaged, "| damaged fractured images (kept, YOLO ignores them):", bad_pos)
    neg, start = {}, 0
    for s in ("train", "val", "test"):
        neg[s], start = negatives[start:start + NEG[s]], start + NEG[s]

    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for s in ("train", "val", "test"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)
        for name, folder in [(n, "Fractured") for n in pos[s]] + [(n, "Non_fractured") for n in neg[s]]:
            shutil.copy(RAW / "images" / folder / name, OUT / "images" / s / name)
            labels = lines.get(name, [])
            (OUT / "labels" / s / name.replace(".jpg", ".txt")).write_text("\n".join(labels))
            rows.append({"image_id": name, "split": s, "fractured": folder == "Fractured", "n_outlines": len(labels)})
    table = pd.DataFrame(rows).merge(info, on="image_id", suffixes=("", "_info"))
    table.to_csv(OUT / "splits.csv", index=False)
    (OUT / "fracatlas.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fracture\n")

    missing = table[table["fractured"] & (table["n_outlines"] == 0)]
    print(table.groupby(["split", "fractured"]).size().unstack().rename(columns={True: "fractured", False: "normal"}))
    print("fractured images without an outline:", len(missing), "| outlines written:", int(table["n_outlines"].sum()))
    print("body parts in test:", table[table["split"] == "test"][["hand", "leg", "hip", "shoulder"]].sum().to_dict())


if __name__ == "__main__":
    main()
