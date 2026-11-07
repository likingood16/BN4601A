"""Tests of image_tools.py on held-out pictures the tools never trained on. Saves outputs/metrics/image_tools_<part>.json.

    python image_tools_check.py --part sorter    # MedGemma's image type vs MultiCaRe's own labels (GPU)
    python image_tools_check.py --part chest     # chest model on the pneumonia test set: AUROC, two sets of weights
    python image_tools_check.py --part outline   # MedSAM outline vs the doctors' fracture outlines (FracAtlas test)

Data is prepared on the HPC login node by hpc/prepare_image_tests.py (data/image_tests/).
"""
import argparse
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

from config import OUTPUTS, PROCESSED, REPO

TESTS = PROCESSED.parent / "image_tests"
FRACATLAS = REPO / "vision" / "detector" / "data" / "yolo"


def save(part, res):
    path = OUTPUTS / "metrics" / f"image_tools_{part}.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1), f"\nsaved {path}")


def check_sorter():
    from image_tools import sort_image
    from model import load
    model, processor = load(mode="4bit")
    labels = pd.read_csv(TESTS / "multicare" / "labels.csv")
    rows = []
    for r in labels.to_dict("records"):
        kind, said = sort_image(model, processor, TESTS / "multicare" / f"{r['file_id']}.jpg")
        rows.append({**r, "predicted": kind, "said": said})
        print(r["true_type"], "->", kind, flush=True)
    t = pd.DataFrame(rows)
    save("sorter", {"n": len(t), "accuracy": float((t["true_type"] == t["predicted"]).mean()),
                    "accuracy_by_type": t.groupby("true_type").apply(lambda g: float((g["true_type"] == g["predicted"]).mean())).to_dict(),
                    "confusion": pd.crosstab(t["true_type"], t["predicted"]).to_dict(), "cases": rows})


def check_chest():
    from image_tools import chest_findings
    test = pd.read_parquet(TESTS / "pneumonia_test.parquet")
    y = test["label"].to_numpy()
    res = {"n": len(test), "n_pneumonia": int(y.sum()), "note": "children's chest X-rays (Kermany 2018): not in either training set"}
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, img in enumerate(test["image"]):
            p = Path(tmp) / f"{i}.png"
            Image.open(io.BytesIO(img["bytes"])).save(p)
            paths.append(p)
        for weights in ("densenet121-res224-all", "densenet121-res224-chex"):
            found = [chest_findings(p, weights) for p in paths]
            for label in ("Pneumonia", "Lung Opacity", "Consolidation"):
                if label in found[0]["scores"]:
                    p = np.array([f["scores"][label] for f in found])
                    res[f"{weights}|{label}"] = {"auroc": float(roc_auc_score(y, p)),
                                                 "sensitivity_at_0.5": float((p[y == 1] >= 0.5).mean()),
                                                 "specificity_at_0.5": float((p[y == 0] < 0.5).mean())}
            print(weights, {k: v for k, v in res.items() if weights in k}, flush=True)
    save("chest", res)


def polygons(label_file, w, h):
    """FracAtlas outlines (YOLO format: class x1 y1 x2 y2 ... in 0-1) -> pixel polygons."""
    out = []
    for line in label_file.read_text().split("\n"):
        v = line.split()
        if len(v) > 6:
            xy = np.array(v[1:], float).reshape(-1, 2) * [w, h]
            out.append([tuple(p) for p in xy])
    return out


def check_outline():
    """Box = the doctors' outline + 10% margin (as if a clinician drew it). Dice 1 = identical outline."""
    from image_tools import outline
    split = pd.read_csv(FRACATLAS / "splits.csv")
    rows = []
    for name in split[(split["split"] == "test") & split["fractured"]]["image_id"]:
        path = FRACATLAS / "images" / "test" / name
        w, h = Image.open(path).size
        for poly in polygons(FRACATLAS / "labels" / "test" / f"{Path(name).stem}.txt", w, h):
            truth = Image.new("L", (w, h))
            ImageDraw.Draw(truth).polygon(poly, fill=1)
            truth = np.asarray(truth, bool)
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            mx, my = 0.1 * (max(xs) - min(xs)) + 5, 0.1 * (max(ys) - min(ys)) + 5
            box = (max(0, min(xs) - mx), max(0, min(ys) - my), min(w, max(xs) + mx), min(h, max(ys) + my))
            mask = outline(path, box)["mask"].astype(bool)
            box_mask = np.zeros_like(truth)
            box_mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
            dice = lambda a: float(2 * (a & truth).sum() / (a.sum() + truth.sum() + 1e-8))
            rows.append({"image": name, "dice_medsam": dice(mask), "dice_box_only": dice(box_mask)})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    save("outline", {"n_outlines": len(t), "median_dice_medsam": float(t["dice_medsam"].median()),
                     "median_dice_box_only": float(t["dice_box_only"].median()),
                     "note": "box_only = just shading the drawn box; MedSAM must beat it to be useful", "cases": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["sorter", "chest", "outline"])
    {"sorter": check_sorter, "chest": check_chest, "outline": check_outline}[ap.parse_args().part]()
