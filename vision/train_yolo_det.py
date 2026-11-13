"""One trainer for the box detectors (fractures, chest X-ray findings): train YOLO11 on a YOLO dataset, pick the
epoch on validation, test ONCE, and report it the way a clinician would use it.

    python train_yolo_det.py --data detector/data/combined/combined.yaml --name fracture_v2
    python train_yolo_det.py --data chest/data/vindr/vindr.yaml --name chest_vindr --imgsz 1024
    add --smoke for 1 epoch on 5% of the data

Test report (runs/<name>/test_metrics.json):
  - YOLO's box mAP50 and mAP50-95, overall and per finding
  - per image, for each finding: AUROC of the top box confidence, and sensitivity/specificity at a cut-off chosen on
    VALIDATION to catch 80% of images with that finding (the same rule as the first fracture detector)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("YOLO_OFFLINE", "true")
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
SEED = 42


def per_image(model, data, split, imgsz, n_cls):
    """[n_images, n_cls] top confidence per finding, and [n_images, n_cls] truth (finding present)."""
    root = Path(data["path"])
    imgs = sorted((root / data[split]).glob("*"))
    conf, truth = np.zeros((len(imgs), n_cls)), np.zeros((len(imgs), n_cls), bool)
    for i, img in enumerate(imgs):
        lab = root / "labels" / split / f"{img.stem}.txt"
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            truth[i, int(line.split()[0])] = True
    # one image per call: with a list of 32 mixed-size X-rays the results came back out of step with the images
    # (8 Nov: fracture AUROC 0.34 batched vs 0.97 one by one on the same 96 test X-rays)
    for i, p in enumerate(imgs):
        r = model.predict(str(p), conf=0.01, imgsz=imgsz, verbose=False)[0]
        for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            conf[i, int(c)] = max(conf[i, int(c)], s)
    return conf, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="skip training; re-test the saved best.pt")
    a = ap.parse_args()
    data_path = (HERE / a.data).resolve()
    data = yaml.safe_load(data_path.read_text())
    names = data["names"] if isinstance(data["names"], list) else [data["names"][k] for k in sorted(data["names"])]
    project = HERE / "runs"
    if not a.score_only:
      YOLO(a.model).train(data=str(data_path), epochs=1 if a.smoke else a.epochs, imgsz=a.imgsz, batch=a.batch,
                fraction=0.05 if a.smoke else 1.0, patience=a.patience, seed=SEED, deterministic=True, workers=a.workers,
                project=str(project), name=a.name + ("_smoke" if a.smoke else ""), exist_ok=True, plots=True, verbose=False)
    run = project / (a.name + ("_smoke" if a.smoke else ""))
    best = YOLO(str(run / "weights" / "best.pt"))
    m = best.val(data=str(data_path), split="test", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)
    res = {"box_mAP50": round(float(m.box.map50), 4), "box_mAP50_95": round(float(m.box.map), 4),
           "mAP50_by_finding": {names[int(c)]: round(float(v), 4) for c, v in zip(m.box.ap_class_index, m.box.ap50)}}
    cv, tv = per_image(best, data, "val", a.imgsz, len(names))
    ct, tt = per_image(best, data, "test", a.imgsz, len(names))
    res["per_image"] = {}
    for k, name in enumerate(names):
        if tt[:, k].sum() < 5 or tv[:, k].sum() < 5:
            res["per_image"][name] = {"n_test_with": int(tt[:, k].sum()), "note": "too few to score"}
            continue
        pos = cv[tv[:, k], k]
        cut = float(np.quantile(pos, 0.2))                        # catches 80% of validation images with it
        reaches = cut > 0
        if not reaches:   # no box at all on >20% of them: a 0 cut-off would flag every image, so use the lowest real score
            cut = float(pos[pos > 0].min()) if (pos > 0).any() else 1.0
        flag = ct[:, k] >= cut
        res["per_image"][name] = {"n_test_with": int(tt[:, k].sum()), "auroc": round(float(roc_auc_score(tt[:, k], ct[:, k])), 3),
                                  "cut_off": round(cut, 4), "reaches_80pct_on_val": bool(reaches),
                                  "sensitivity": round(float(flag[tt[:, k]].mean()), 3),
                                  "specificity": round(float((~flag[~tt[:, k]]).mean()), 3)}
    res.update({"model": a.model, "imgsz": a.imgsz, "smoke": a.smoke, "names": names})
    (run / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
