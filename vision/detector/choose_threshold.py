"""Choose the detector's confidence cut-off on VALIDATION X-rays, then report it once on test (CPU, a few minutes).

    python choose_threshold.py                                  # runs/fracture, 640 px, CPU, 80% target
    python choose_threshold.py --run fracture_m1024 --imgsz 1024 --device 0

Training used conf 0.25 (Ultralytics default), which caught 54% of fractured test X-rays. A lower cut-off catches
more fractures and marks more normal X-rays. Per X-ray: fractured if any outline reaches the cut-off.
Saves runs/fracture/threshold.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("YOLO_OFFLINE", "true")
from ultralytics import YOLO  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data" / "yolo"
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="fracture")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="cpu")
ap.add_argument("--target", type=float, default=0.80)
A = ap.parse_args()
RUN = HERE / "runs" / A.run
TARGET = A.target


def top_conf(model, split):
    """Highest outline confidence per X-ray (0 if nothing at all above 0.01)."""
    rows = pd.read_csv(DATA / "splits.csv").query("split == @split")
    tops = []
    for name in rows["image_id"]:
        r = model.predict(str(DATA / "images" / split / name), conf=0.01, imgsz=A.imgsz, device=A.device, verbose=False)[0]
        tops.append(max(r.boxes.conf.tolist(), default=0.0))
    return rows["fractured"].to_numpy(bool), np.array(tops)


def rates(y, top, cut):
    flag = top >= cut
    return {"sensitivity": round(float(flag[y].mean()), 3), "specificity": round(float(1 - flag[~y].mean()), 3)}


model = YOLO(str(RUN / "weights" / "best.pt"))
y_va, top_va = top_conf(model, "val")
y_te, top_te = top_conf(model, "test")
cut = float(np.quantile(top_va[y_va], 1 - TARGET))           # catches TARGET of fractured validation X-rays
res = {"target_sensitivity": TARGET, "chosen_cut_off": round(cut, 3),
       "val": rates(y_va, top_va, cut), "test": rates(y_te, top_te, cut),
       "test_at_0.25 (old)": rates(y_te, top_te, 0.25),
       "test_auroc_per_xray": round(float(roc_auc_score(y_te, top_te)), 3),   # cut-off free: compare runs with this
       "val_curve": {f"{c:.2f}": rates(y_va, top_va, c) for c in (0.05, 0.1, 0.15, 0.2, 0.25)}}
(RUN / "threshold.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
