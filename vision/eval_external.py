"""Extra tests for the four image models on pictures they never trained on (8 Nov 2026).

    python eval_external.py                       # GPU job: PART=external in reasoning/hpc/jobs/train_images.pbs
    python eval_external.py --parts imagetype     # only some parts; each part is skipped if its model isn't trained yet

1. Fracture detector (trained on GRAZPEDWRI-DX children's wrists + FracAtlas):
   - its own test set split by source: GRAZ (children, wrist) vs FracAtlas (adults, hand/leg/hip/shoulder)
   - MultiCaRe bone X-rays from published case reports: caption mentions a fracture or not (weak labels)
2. Chest detector (trained on VinDr-CXR, adults, Vietnam):
   - pneumonia test set (624 children's chest X-rays, Guangzhou; 390 pneumonia): pneumonia score = highest of
     consolidation / lung opacity / infiltration
   - MultiCaRe chest X-rays: caption mentions pneumothorax / effusion / cardiomegaly / consolidation-pneumonia
3. Head-CT bleed (trained on RSNA 2019 slices): per PATIENT (any slice above the cut-off), which is how a scan is read;
   MultiCaRe head CT pictures whose caption mentions a bleed
4. Image-type sorter (trained on MultiCaRe): the 280 held-out MultiCaRe pictures, plus 200 pictures from each other
   dataset (VinDr, GRAZ, FracAtlas, RSNA head CT, pneumonia set). If it sorts these wrong, the right model never runs.
Cut-offs always come from each model's VALIDATION set (never from these tests). Saves runs/external_tests.json.
"""
import argparse
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TESTS = REPO / "reasoning" / "data" / "image_tests"
OUT = HERE / "runs" / "external_tests.json"
SEED = 42


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return round(float(roc_auc_score(y, s)), 3) if 0 < y.sum() < len(y) else None


def rates(y, flag):
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    return {"n": int(len(y)), "n_with": int(y.sum()),
            "sensitivity": round(float(flag[y].mean()), 3) if y.any() else None,
            "specificity": round(float((~flag[~y]).mean()), 3) if (~y).any() else None}


def top_scores(model, paths, imgsz, n_cls):
    """[n, n_cls] highest box confidence per class, one image per call (batched calls came back out of step)."""
    out = np.zeros((len(paths), n_cls))
    for i, p in enumerate(paths):
        r = model.predict(p, conf=0.01, imgsz=imgsz, verbose=False)[0]
        for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            out[i, int(c)] = max(out[i, int(c)], s)
    return out


def val_cut(model, root, imgsz, n_cls, k, frac=0.2):
    """Cut-off catching 80% of validation images with class k (lowest real score if 80% isn't reachable)."""
    imgs = sorted((root / "images" / "val").glob("*"))
    truth = np.array([any(l.split()[0] == str(k) for l in (root / "labels" / "val" / f"{p.stem}.txt").read_text().splitlines())
                      if (root / "labels" / "val" / f"{p.stem}.txt").exists() else False for p in imgs])
    s = top_scores(model, [str(p) for p in imgs], imgsz, n_cls)[:, k]
    pos = s[truth]
    cut = float(np.quantile(pos, frac))
    return cut if cut > 0 else float(pos[pos > 0].min())


def multicare():
    lab = pd.read_csv(TESTS / "multicare" / "labels.csv")
    lab["path"] = [str(TESTS / "multicare" / f"{f}.jpg") for f in lab["file_id"]]
    lab["caption"] = lab["caption"].fillna("").str.lower()
    return lab


def pneumonia_set():
    from PIL import Image
    df = pd.read_parquet(TESTS / "pneumonia_test.parquet")
    folder = HERE / "runs" / "_pneumonia_png"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(df["image"]):
        p = folder / f"{i:04d}.png"
        if not p.exists():
            Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img)).convert("RGB").save(p)
        paths.append(str(p))
    return paths, df["label"].to_numpy().astype(bool)


def fracture(res):
    from ultralytics import YOLO
    w = HERE / "runs" / "fracture_v2" / "weights" / "best.pt"
    if not w.exists():
        return
    m, root = YOLO(str(w)), HERE / "detector" / "data" / "combined"
    cut = val_cut(m, root, 640, 2, 0)
    imgs = sorted((root / "images" / "test").glob("*"))
    truth = np.array([any(l.split()[0] == "0" for l in (root / "labels" / "test" / f"{p.stem}.txt").read_text().splitlines())
                      if (root / "labels" / "test" / f"{p.stem}.txt").exists() else False for p in imgs])
    s = top_scores(m, [str(p) for p in imgs], 640, 2)[:, 0]
    src = np.array(["FracAtlas (adults, many body parts)" if p.name.startswith("fa_") else "GRAZ (children, wrist)" for p in imgs])
    out = {"cut_off_from_val": round(cut, 4), "all": {"auroc": auroc(truth, s), **rates(truth, s >= cut)}}
    for name in np.unique(src):
        k = src == name
        out[name] = {"auroc": auroc(truth[k], s[k]), **rates(truth[k], s[k] >= cut)}
    mc = multicare()
    mc = mc[mc["true_type"] == "bone_xray"]
    y = mc["caption"].str.contains(r"fractur").to_numpy() & ~mc["caption"].str.contains(r"\bno (?:\w+ )?fractur").to_numpy()
    s2 = top_scores(m, mc["path"].tolist(), 640, 2)[:, 0]
    out["MultiCaRe bone X-rays (caption says fracture)"] = {"auroc": auroc(y, s2), **rates(y, s2 >= cut)}
    res["fracture"] = out


def fracture_adult(res, run="fracture_v2"):
    """8 Nov: at the shared cut-off adults (FracAtlas) were caught 22% vs children's wrists 84%, because the cut-off was
    set mostly by GRAZ. Here the cut-off is set on ADULT validation X-rays only (catch 90%), then tested on both."""
    from ultralytics import YOLO
    w = HERE / "runs" / run / "weights" / "best.pt"
    if not w.exists():
        return
    m, root = YOLO(str(w)), HERE / "detector" / "data" / "combined"

    def scored(split, prefix):
        imgs = sorted((root / "images" / split).glob(f"{prefix}*"))
        y = np.array([any(l.split()[0] == "0" for l in (root / "labels" / split / f"{p.stem}.txt").read_text().splitlines())
                      if (root / "labels" / split / f"{p.stem}.txt").exists() else False for p in imgs])
        return y, top_scores(m, [str(p) for p in imgs], 640, 2)[:, 0]
    yv, sv = scored("val", "fa_")
    cut = float(np.quantile(sv[yv], 0.1))
    cut = cut if cut > 0 else float(sv[yv][sv[yv] > 0].min())
    out = {"cut_off_from_adult_val": round(cut, 4), "adult_val_with": int(yv.sum())}
    for name, prefix in [("FracAtlas (adults)", "fa_"), ("GRAZ (children, wrist)", "graz_")]:
        y, s = scored("test", prefix)
        out[name] = {"auroc": auroc(y, s), **rates(y, s >= cut)}
    res[f"fracture_adult_cut__{run}"] = out


CHEST = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
         "Lung opacity", "Nodule/mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
         "Pulmonary fibrosis"]


def chest(res):
    from ultralytics import YOLO
    w = HERE / "runs" / "chest_vindr" / "weights" / "best.pt"
    if not w.exists():
        return
    m, root = YOLO(str(w)), HERE / "chest" / "data" / "vindr"
    pneu = [CHEST.index(c) for c in ("Consolidation", "Lung opacity", "Infiltration")]
    cuts = {k: val_cut(m, root, 1024, 14, k) for k in pneu + [CHEST.index(c) for c in ("Pneumothorax", "Pleural effusion", "Cardiomegaly")]}
    paths, y = pneumonia_set()
    s = top_scores(m, paths, 1024, 14)
    flag = np.any([s[:, k] >= cuts[k] for k in pneu], axis=0)
    out = {"pneumonia set (children, 624)": {"auroc_highest_of_3": auroc(y, s[:, pneu].max(1)), **rates(y, flag),
                                              **{f"auroc_{CHEST[k]}": auroc(y, s[:, k]) for k in pneu}}}
    mc = multicare()
    mc = mc[mc["true_type"] == "chest_xray"]
    s2 = top_scores(m, mc["path"].tolist(), 1024, 14)
    for name, rule in [("Pneumothorax", r"pneumothora"), ("Pleural effusion", r"effusion"), ("Cardiomegaly", r"cardiomegal|enlarged heart"),
                       ("Consolidation", r"consolidat|pneumonia")]:
        k = CHEST.index(name)
        yk = mc["caption"].str.contains(rule).to_numpy()
        ks = [k] if name != "Consolidation" else pneu
        sk = s2[:, ks].max(1)
        out[f"MultiCaRe chest X-rays: {name} in caption"] = {"auroc": auroc(yk, sk),
                                                             **rates(yk, np.any([s2[:, j] >= cuts[j] for j in ks], axis=0))}
    out["cut_offs_from_val"] = {CHEST[k]: round(v, 4) for k, v in cuts.items()}
    # The app's other chest model (TorchXRayVision DenseNet, CheXpert weights; whole-image scores, no boxes) on the
    # same pictures: if it is better at "is it there?", the app can use it for that and the detector for the box.
    xs = {"pneumonia set": xrv_scores(paths), "MultiCaRe": xrv_scores(mc["path"].tolist())}
    vindr_val = sorted((root / "images" / "val").glob("*"))
    vindr_test = sorted((root / "images" / "test").glob("*"))
    xv, xt = xrv_scores([str(p) for p in vindr_val]), xrv_scores([str(p) for p in vindr_test])

    def has(split, imgs, names):
        ks = {str(CHEST.index(n)) for n in names}
        return np.array([any(l.split()[0] in ks for l in (root / "labels" / split / f"{p.stem}.txt").read_text().splitlines())
                         for p in imgs])
    compare = {}
    for xname, vnames in [("Pneumothorax", ["Pneumothorax"]), ("Effusion", ["Pleural effusion"]), ("Cardiomegaly", ["Cardiomegaly"]),
                          ("Pneumonia", ["Consolidation", "Lung opacity", "Infiltration"])]:
        yv, yt = has("val", vindr_val, vnames), has("test", vindr_test, vnames)
        cut = float(np.quantile(xv[xname][yv], 0.2))                       # catches 80% on VinDr validation
        compare[xname] = {"cut_off_from_vindr_val": round(cut, 4),
                          "VinDr test": {"auroc": auroc(yt, xt[xname]), **rates(yt, xt[xname] >= cut)}}
        if xname == "Pneumonia":
            compare[xname]["pneumonia set"] = {"auroc": auroc(y, xs["pneumonia set"][xname]),
                                               **rates(y, xs["pneumonia set"][xname] >= cut)}
        rule = {"Pneumothorax": r"pneumothora", "Effusion": r"effusion", "Cardiomegaly": r"cardiomegal|enlarged heart",
                "Pneumonia": r"consolidat|pneumonia"}[xname]
        yk = mc["caption"].str.contains(rule).to_numpy()
        compare[xname]["MultiCaRe"] = {"auroc": auroc(yk, xs["MultiCaRe"][xname]), **rates(yk, xs["MultiCaRe"][xname] >= cut)}
    out["CheXpert classifier (no boxes), same tests"] = compare
    res["chest"] = out


def xrv_scores(paths):
    """TorchXRayVision CheXpert-weights scores per picture, as the app computes them (reasoning/src/image_tools.py)."""
    import sys
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPO / "reasoning" / "src"))
    from image_tools import CHEST_WEIGHTS, _import_xrv
    xrv = _import_xrv()
    net = xrv.models.DenseNet(weights=CHEST_WEIGHTS).eval()
    rows = []
    with torch.no_grad():
        for p in paths:
            img = xrv.datasets.normalize(np.asarray(Image.open(p).convert("L"), np.float32), 255)[None]
            img = xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(img))
            rows.append(net(torch.from_numpy(img)[None])[0].numpy())
    rows = np.array(rows)
    return {n: rows[:, i] for i, n in enumerate(net.pathologies) if n}


def _effnet(path, n):
    import torch
    from torchvision.models import efficientnet_b0
    m = efficientnet_b0()
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def _run(model, paths, tf, sigmoid, batch=64):
    import torch
    from PIL import Image
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]).to(dev)
            y = model(x)
            out.append((torch.sigmoid(y) if sigmoid else torch.softmax(y, 1)).cpu().numpy())
    return np.concatenate(out)


def ich(res):
    from torchvision import transforms as T
    run = HERE / "ct_head" / "runs" / "ich"
    if not (run / "model.pt").exists() or not (run / "test_metrics.json").exists():
        return
    labels = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    m = _effnet(run / "model.pt", 6)
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cut = json.loads((run / "test_metrics.json").read_text())["any_cut_off_90pct_val"]
    df = pd.read_csv(run / "slices.csv")
    te = df[df["split"] == "test"].reset_index(drop=True)
    p = _run(m, te["path"].tolist(), tf, True)
    te["score"] = p[:, 0]
    per = te.groupby("PatientID").agg(bleed=("any", "max"), score=("score", "max"),
                                      **{l: (l, "max") for l in labels[1:]})
    out = {"slice_cut_off_from_val": cut,
           "per_patient": {"auroc": auroc(per["bleed"], per["score"]), **rates(per["bleed"], per["score"] >= cut)},
           "per_patient_caught_by_type": {l: round(float((per.loc[per[l] == 1, "score"] >= cut).mean()), 3)
                                          for l in labels[1:] if (per[l] == 1).any()}}
    mc = multicare()
    mc = mc[(mc["true_type"] == "ct_slice") & mc["radiology_region"].fillna("").str.contains("head|brain|skull", case=False)]
    if len(mc):
        y = mc["caption"].str.contains(r"h(?:a)?emorrhag|h(?:a)?ematoma|bleed").to_numpy()
        s = _run(m, mc["path"].tolist(), tf, True)[:, 0]
        out["MultiCaRe head CT (caption says bleed)"] = {"auroc": auroc(y, s), **rates(y, s >= cut)}
    res["ich"] = out


def imagetype(res):
    from torchvision import transforms as T
    w = HERE / "image_type" / "runs" / "image_type" / "model.pt"
    if not w.exists():
        return
    types = ["bone_xray", "chest_xray", "ct_slice", "mri_slice", "ultrasound", "ecg", "other"]
    m = _effnet(w, len(types))
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    mc = multicare()
    pred = [types[i] for i in _run(m, mc["path"].tolist(), tf, False).argmax(1)]
    out = {"MultiCaRe 280 held out": {"accuracy": round(float(np.mean(np.array(pred) == mc["true_type"].to_numpy())), 3),
                                      "by_type": {t: round(float(np.mean(np.array(pred)[mc["true_type"] == t] == t)), 3)
                                                  for t in types if (mc["true_type"] == t).any()},
                                      "confusion": pd.crosstab(mc["true_type"], pd.Series(pred, name="pred")).to_dict()}}
    rng = random.Random(SEED)
    comb = HERE / "detector" / "data" / "combined" / "images" / "test"
    sets = {"VinDr chest X-rays": ("chest_xray", sorted((HERE / "chest" / "data" / "vindr" / "images" / "test").glob("*"))),
            "GRAZ wrist X-rays": ("bone_xray", sorted(comb.glob("graz_*"))),
            "FracAtlas X-rays": ("bone_xray", sorted(comb.glob("fa_*"))),
            "pneumonia set chest X-rays": ("chest_xray", [Path(p) for p in pneumonia_set()[0]])}
    run = HERE / "ct_head" / "runs" / "ich" / "slices.csv"
    if run.exists():
        sets["RSNA head CT slices"] = ("ct_slice", [Path(p) for p in pd.read_csv(run).query("split == 'test'")["path"]])
    for name, (want, paths) in sets.items():
        paths = rng.sample(paths, min(200, len(paths)))
        got = [types[i] for i in _run(m, [str(p) for p in paths], tf, False).argmax(1)]
        out[name] = {"want": want, "n": len(got), "correct": round(float(np.mean(np.array(got) == want)), 3),
                     "sorted_as": pd.Series(got).value_counts().to_dict()}
    res["imagetype"] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="imagetype,fracture,chest,ich")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for part in a.parts.split(","):
        print("==", part, flush=True)
        name, _, run = part.partition(":")                 # e.g. fracture_adult:fracture_v3
        globals()[name](res, run) if run else globals()[name](res)
        part = f"{name}_cut__{run}" if name == "fracture_adult" and run else (
            "fracture_adult_cut__fracture_v2" if name == "fracture_adult" else name)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res.get(part, "model not trained yet: skipped"), indent=1, default=str)[:3000], flush=True)
