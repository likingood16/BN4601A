"""Fracture detector data, bigger: GRAZPEDWRI-DX (20,327 paediatric wrist X-rays) + FracAtlas (adult limbs, hip,
shoulder) as one YOLO detection dataset. Run on the HPC login node after download_image_data.sh.

    python prepare_combined.py

GRAZPEDWRI-DX (Nagy et al. 2022, Sci Data, CC BY 4.0): YOLO boxes for 9 classes; we keep "fracture" (class 0 here)
and "foreignbody" (class 1, also an ED question). Split 70/15/15 BY PATIENT (patient_id in dataset.csv), seed 42.
FracAtlas: the same official split as before; its fracture outlines become boxes (class 0). Its normal X-rays as
before (574/82/400). So the FracAtlas TEST set is unchanged and the new detector compares directly with the old one
(box mAP50 0.379, per-X-ray AUROC 0.856, sensitivity 0.81 at the 0.044 cut-off).
Images are linked, not copied. Writes data/combined/{images,labels}/{train,val,test} and combined.yaml.
"""
import os
import random
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
GRAZ = Path("/scratch/e0959158/bn4601a/image_data/grazpedwri")
FRAC = HERE / "data" / "yolo"
OUT = HERE / "data" / "combined"
KEEP = {"fracture": 0, "foreignbody": 1}
SEED = 42


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src, dst)


def graz():
    meta = yaml.safe_load(next(GRAZ.glob("unpacked/**/yolov5/meta.yaml")).read_text())
    names = meta["names"] if isinstance(meta["names"], list) else [meta["names"][k] for k in sorted(meta["names"])]
    label_dir = next(GRAZ.glob("unpacked/**/yolov5/labels"))
    pngs = {p.stem: p for p in GRAZ.glob("unpacked/**/*.png")}
    table = pd.read_csv(GRAZ / "dataset.csv", encoding="utf-8-sig")
    patients = sorted(table["patient_id"].unique())
    random.Random(SEED).shuffle(patients)
    n = len(patients)
    split_of = {p: "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test" for i, p in enumerate(patients)}
    counts = {}
    for r in table.itertuples():
        if r.filestem not in pngs:
            continue
        split = split_of[r.patient_id]
        lines = []
        lab = label_dir / f"{r.filestem}.txt"
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            cls, *box = line.split()
            name = names[int(cls)]
            if name in KEEP:
                lines.append(" ".join([str(KEEP[name]), *box]))
        link(pngs[r.filestem], OUT / "images" / split / f"graz_{r.filestem}.png")
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split / f"graz_{r.filestem}.txt").write_text("\n".join(lines))
        key = (split, "fracture" if any(l.startswith("0 ") for l in lines) else "no fracture")
        counts[key] = counts.get(key, 0) + 1
    print("GRAZPEDWRI-DX:", dict(sorted(counts.items())), "| classes in source:", names, flush=True)
    return split_of


def fracatlas():
    """FracAtlas outlines (YOLO-seg polygons) -> boxes."""
    counts = {}
    for split in ("train", "val", "test"):
        for img in (FRAC / "images" / split).glob("*"):
            lines = []
            seg = FRAC / "labels" / split / f"{img.stem}.txt"
            for line in (seg.read_text().splitlines() if seg.exists() else []):
                v = [float(x) for x in line.split()[1:]]
                xs, ys = v[0::2], v[1::2]
                if xs:
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    lines.append(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
            link(img.resolve(), OUT / "images" / split / f"fa_{img.name}")
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split / f"fa_{img.stem}.txt").write_text("\n".join(lines))
            counts[split] = counts.get(split, 0) + 1
    print("FracAtlas:", counts, flush=True)


if __name__ == "__main__":
    graz()
    fracatlas()
    (OUT / "combined.yaml").write_text(yaml.safe_dump({
        "path": str(OUT), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {v: k for k, v in KEEP.items()}}))
    print("wrote", OUT / "combined.yaml")
