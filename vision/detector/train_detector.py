"""Train + test a fracture detector that also draws outlines (YOLO11 segmentation) on FracAtlas.

    python train_detector.py --epochs 60                 # train (laptop GPU), best epoch chosen on validation
    python train_detector.py --smoke                      # 1 epoch on 5% of the data: check everything runs
    python train_detector.py --test-only runs/fracture/weights/best.pt

Test (once, at the end), two kinds of numbers:
  1. YOLO's standard detection scores on the test split: mAP50 and mAP50-95 for boxes AND outlines (masks).
     mAP50 = how well boxes overlap the radiologist's (at least 50% overlap counts as a hit), averaged over confidence.
  2. Per X-ray, as a clinician would use it: an image is "flagged" if the detector marks any fracture.
     sensitivity = fractured X-rays flagged, specificity = normal X-rays left unflagged (400 normal test images).
Saves runs/<name>/test_metrics.json and a few example images with drawn outlines.
"""
import argparse
import json
import os
from pathlib import Path

# Windows + conda: torch and numpy each bring an OpenMP library; without this the run aborts (OMP Error #15)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "yolo" / "fracatlas.yaml"
SEED = 42


def train(args):
    run = HERE / "runs" / args.name
    if args.resume:                             # continue an interrupted run from its last saved epoch
        YOLO(str(run / "weights" / "last.pt")).train(resume=True)
        return run / "weights" / "best.pt"
    model = YOLO(args.model)                    # pretrained on COCO everyday images; we fine-tune on X-rays
    model.train(data=str(DATA), epochs=1 if args.smoke else args.epochs, imgsz=args.imgsz, batch=args.batch,
                fraction=0.05 if args.smoke else 1.0, patience=args.patience, seed=SEED, deterministic=True,
                workers=0, project=str(HERE / "runs"), name=args.name + ("_smoke" if args.smoke else ""),
                exist_ok=True, plots=True, verbose=False)
    return HERE / "runs" / (args.name + ("_smoke" if args.smoke else "")) / "weights" / "best.pt"


def test(weights, conf):
    model = YOLO(str(weights))
    m = model.val(data=str(DATA), split="test", imgsz=640, batch=8, plots=False, verbose=False)
    scores = {"box_mAP50": m.box.map50, "box_mAP50_95": m.box.map, "mask_mAP50": m.seg.map50,
              "mask_mAP50_95": m.seg.map, "box_precision": m.box.mp, "box_recall": m.box.mr}

    split = pd.read_csv(HERE / "data" / "yolo" / "splits.csv")
    test_rows = split[split["split"] == "test"]
    img_dir = HERE / "data" / "yolo" / "images" / "test"
    flagged = {}
    for n in test_rows["image_id"]:              # one X-ray at a time: a list would be sent as ONE batch (4 GB GPU)
        flagged[n] = len(model.predict(str(img_dir / n), conf=conf, imgsz=640, verbose=False)[0].boxes) > 0
    test_rows = test_rows.assign(flagged=test_rows["image_id"].map(flagged))
    frac, normal = test_rows[test_rows["fractured"]], test_rows[~test_rows["fractured"]]
    scores.update({"confidence_threshold": conf,
                   "per_xray_sensitivity": frac["flagged"].mean(), "per_xray_specificity": 1 - normal["flagged"].mean(),
                   "n_fractured_test": len(frac), "n_normal_test": len(normal)})
    for part in ("hand", "leg", "hip", "shoulder"):
        sub = frac[frac[part] == 1]
        if len(sub):
            scores[f"sensitivity_{part}"] = sub["flagged"].mean()
    scores = {k: round(float(v), 4) if isinstance(v, float) else int(v) for k, v in scores.items()}

    run_dir = Path(weights).resolve().parent.parent      # absolute, or YOLO nests it under runs/segment/
    (run_dir / "test_metrics.json").write_text(json.dumps(scores, indent=1))
    examples = frac["image_id"].head(4).tolist() + normal["image_id"].head(2).tolist()
    for n in examples:
        model.predict(str(img_dir / n), conf=conf, save=True, project=str(run_dir), name="test_examples",
                      exist_ok=True, verbose=False)
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-seg.pt", help="small segmentation model, fits a 4 GB laptop GPU")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15, help="stop when validation stops improving")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="fracture")
    ap.add_argument("--conf", type=float, default=0.25, help="detections below this confidence are ignored")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from its last saved epoch")
    ap.add_argument("--test-only", default=None, help="weights to test without training")
    a = ap.parse_args()
    weights = a.test_only or train(a)
    print(json.dumps(test(weights, a.conf), indent=1))
