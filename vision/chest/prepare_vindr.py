"""Chest X-ray findings with boxes: VinDr-CXR (VinBigData Kaggle competition, 15,000 X-rays, 14 findings, each read
by 3 radiologists) as a YOLO detection dataset. Run on the HPC login node after download_kaggle.sh.

    python prepare_vindr.py

Boxes: train.csv gives every radiologist's boxes in original pixels. For each X-ray and finding, boxes from different
radiologists that overlap (IoU > 0.3) are merged into one (their average), so one finding = one box. X-rays all three
called "No finding" have an empty label (the detector also learns what normal looks like).
Coordinates are divided by the original width/height (train_meta.csv), so they fit the resized 1024 px PNGs.
Split 70/15/15 by X-ray, seed 42 (VinDr gives no patient IDs; each study is one patient visit).
Data stays on vanda (competition terms: non-commercial, not shared). Writes data/vindr/{images,labels}/... + vindr.yaml.
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SRC = Path("/scratch/e0959158/bn4601a/image_data/vinbigdata")
OUT = HERE / "data" / "vindr"
NAMES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
         "Lung opacity", "Nodule/mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
         "Pulmonary fibrosis"]                       # VinDr class_id 0-13; 14 = No finding
SEED = 42


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def merge(boxes):
    """Group overlapping boxes from different radiologists, average each group."""
    groups = []
    for b in boxes:
        for g in groups:
            if iou(np.mean(g, 0), b) > 0.3:
                g.append(b)
                break
        else:
            groups.append([b])
    return [np.mean(g, 0) for g in groups]


def main():
    train = pd.read_csv(SRC / "train.csv")
    meta = pd.read_csv(next(SRC.rglob("train_meta.csv"))).set_index("image_id")      # dim0 = height, dim1 = width
    pngs = {p.stem: p for p in SRC.rglob("train/*.png")}
    ids = sorted(set(train["image_id"]) & set(pngs) & set(meta.index))
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    split_of = {i: "train" if k < 0.7 * n else "val" if k < 0.85 * n else "test" for k, i in enumerate(ids)}
    found = train[train["class_id"] < 14]
    counts = {s: np.zeros(len(NAMES), int) for s in ("train", "val", "test")}
    normal = {s: 0 for s in counts}
    for image_id in ids:
        split = split_of[image_id]
        h, w = meta.loc[image_id, "dim0"], meta.loc[image_id, "dim1"]
        lines = []
        for cls, rows in found[found["image_id"] == image_id].groupby("class_id"):
            for x0, y0, x1, y1 in merge(rows[["x_min", "y_min", "x_max", "y_max"]].to_numpy(float).tolist()):
                lines.append(f"{int(cls)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
                counts[split][int(cls)] += 1
        normal[split] += not lines
        dst = OUT / "images" / split / f"{image_id}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(pngs[image_id], dst)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines))
    table = pd.DataFrame(counts, index=NAMES)
    print(table.to_string(), "\nnormal X-rays:", normal, "\ntotal X-rays:", n, flush=True)
    (OUT / "vindr.yaml").write_text(yaml.safe_dump({"path": str(OUT), "train": "images/train", "val": "images/val",
                                                    "test": "images/test", "names": dict(enumerate(NAMES))}))


if __name__ == "__main__":
    main()
