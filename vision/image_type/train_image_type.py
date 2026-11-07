"""Image-type classifier: which kind of picture is this (bone X-ray, chest X-ray, CT, MRI, ultrasound, ECG, other)?
It decides which detector/segmenter runs next, so it replaces MedGemma's guess (9/10 in the real-AI check; MRI read
as CT).

    python train_image_type.py --prepare      # login node: export MultiCaRe pictures per type, split by article
    python train_image_type.py                # GPU: train, pick the epoch on validation, test once
    python train_image_type.py --smoke        # 1 epoch on 3% of the data

Data: MultiCaRe images (OpenMed/multicare-images, CC BY), labelled by MultiCaRe's own image_subtype and
radiology_region (same mapping as reasoning/hpc/prepare_image_tests.py). Split 70/15/15 BY ARTICLE (one case report's
pictures stay together), seed 42, at most 3,000 pictures per type so no type swamps the others. The 280 pictures
used in the earlier image tests are left out of training.
Model: EfficientNet-B0 pretrained on ImageNet (torchvision), all layers fine-tuned, 224 px.
Saves runs/image_type/{model.pt, test_metrics.json} (accuracy, per-type sensitivity, confusion matrix).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path("/scratch/e0959158/bn4601a/image_data/image_type")
RUN = HERE / "runs" / "image_type"
TYPES = ["bone_xray", "chest_xray", "ct_slice", "mri_slice", "ultrasound", "ecg", "other"]
OTHER = {"pathology", "medical_photograph", "endoscopy", "chart", "ophthalmic_imaging"}
PER_TYPE = 3000
SEED = 42


def our_type(r):
    sub, region = r["image_subtype"], r["radiology_region"]
    if sub == "x_ray":
        return {"thorax": "chest_xray", "upper_limb": "bone_xray", "lower_limb": "bone_xray", "pelvis": "bone_xray"}.get(region)
    if sub in ("ct", "mri", "ultrasound"):
        return {"ct": "ct_slice", "mri": "mri_slice", "ultrasound": "ultrasound"}[sub]
    if sub == "ekg":
        return "ecg"
    return "other" if r["image_type"] in OTHER else None


def prepare():
    from datasets import load_dataset
    held_out = set(pd.read_csv(REPO / "reasoning" / "data" / "image_tests" / "multicare" / "labels.csv")["file_id"])
    ds = load_dataset("OpenMed/multicare-images", split="train")
    meta = ds.remove_columns([c for c in ds.column_names if c == "image"]).to_pandas()
    meta["kind"] = [our_type(r) for r in meta.to_dict("records")]
    meta = meta[meta["kind"].notna() & ~meta["file_id"].isin(held_out)]
    # group by patient when MultiCaRe gives one (it does: patient_id), else by case report
    meta["article"] = meta["patient_id"].astype(str) if "patient_id" in meta else meta["file_id"].str.split("_").str[0]
    rng = random.Random(SEED)
    articles = sorted(meta["article"].unique())
    rng.shuffle(articles)
    n = len(articles)
    split_of = {a: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test") for i, a in enumerate(articles)}
    meta["split"] = meta["article"].map(split_of)
    keep = meta.sample(frac=1, random_state=SEED).groupby("kind").head(PER_TYPE)   # at most PER_TYPE per type, random
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    for row_i, r in zip(keep.index, keep.to_dict("records")):
        out = DATA / "images" / f"{r['file_id']}.jpg"
        if not out.exists():
            ds[int(row_i)]["image"].convert("RGB").save(out, quality=90)
    keep[["file_id", "kind", "split", "article", "image_subtype", "radiology_region"]].to_csv(DATA / "labels.csv", index=False)
    print(keep.groupby(["kind", "split"]).size().unstack(), flush=True)


HOSPITAL = {   # real hospital pictures (8 Nov): MultiCaRe alone is journal figures (cropped, arrows, labels)
    "vindr": ("chest_xray", lambda s: sorted((REPO / "vision/chest/data/vindr/images" / s).glob("*"))),
    "graz": ("bone_xray", lambda s: sorted((REPO / "vision/detector/data/combined/images" / s).glob("graz_*"))),
    "fracatlas": ("bone_xray", lambda s: sorted((REPO / "vision/detector/data/combined/images" / s).glob("fa_*"))),
    "rsna_ct": ("ct_slice", lambda s: [Path(p) for p in pd.read_csv(REPO / "vision/ct_head/runs/ich/slices.csv")
                                      .query("split == @s")["path"]]),
}
HOSPITAL_N = {"train": 700, "val": 150, "test": 150}


def add_hospital():
    """Add up to 700/150/150 pictures per hospital dataset, each from that dataset's OWN split (by patient or X-ray),
    so no test patient of those datasets is trained on. Writes labels_v2.csv (MultiCaRe rows + these)."""
    labels = pd.read_csv(DATA / "labels.csv")
    labels["source"], labels["path"] = "multicare", [str(DATA / "images" / f"{f}.jpg") for f in labels["file_id"]]
    rows, rng = [labels], random.Random(SEED)
    for source, (kind, files) in HOSPITAL.items():
        for split, n in HOSPITAL_N.items():
            paths = files(split)
            paths = rng.sample(paths, min(n, len(paths)))
            rows.append(pd.DataFrame({"file_id": [f"{source}_{p.stem}" for p in paths], "kind": kind, "split": split,
                                      "source": source, "path": [str(p) for p in paths]}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA / "labels_v2.csv", index=False)
    print(out.groupby(["source", "split"]).size().unstack(), flush=True)


def loaders(smoke, batch=64):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    labels = pd.read_csv(DATA / "labels_v2.csv") if (DATA / "labels_v2.csv").exists() else pd.read_csv(DATA / "labels.csv")
    if "path" not in labels:
        labels["path"] = [str(DATA / "images" / f"{f}.jpg") for f in labels["file_id"]]
    if smoke:
        labels = labels.sample(frac=0.03, random_state=SEED)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = {"train": T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomRotation(10), T.ColorJitter(0.2, 0.2),
                              T.ToTensor(), norm]),
          "eval": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])}

    class Pics(torch.utils.data.Dataset):
        def __init__(self, rows, t):
            self.rows, self.t = rows.reset_index(drop=True), t

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows.iloc[i]
            return self.t(Image.open(r.path).convert("RGB")), TYPES.index(r.kind)
    out = {"test_rows": labels[labels["split"] == "test"].reset_index(drop=True)}
    for split in ("train", "val", "test"):
        rows = labels[labels["split"] == split]
        out[split] = torch.utils.data.DataLoader(Pics(rows, tf["train" if split == "train" else "eval"]), batch_size=batch,
                                                 shuffle=split == "train", num_workers=8)
    return out


def evaluate(model, loader, device):
    import torch
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train(smoke, epochs):
    import torch
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(TYPES))
    model = model.to(device)
    data = loaders(smoke)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    RUN.mkdir(parents=True, exist_ok=True)
    best, history = -1, []
    for epoch in range(1, (1 if smoke else epochs) + 1):
        model.train()
        for x, y in data["train"]:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=0.05)
            loss.backward()
            opt.step()
        sched.step()
        y, p = evaluate(model, data["val"], device)
        acc = float((y == p).mean())
        history.append({"epoch": epoch, "val_accuracy": round(acc, 4)})
        print(history[-1], flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), RUN / "model.pt")
    model.load_state_dict(torch.load(RUN / "model.pt", map_location=device))
    y, p = evaluate(model, data["test"], device)
    cm = pd.crosstab(pd.Series([TYPES[i] for i in y], name="true"), pd.Series([TYPES[i] for i in p], name="predicted"))
    res = {"test_n": int(len(y)), "test_accuracy": round(float((y == p).mean()), 4),
           "sensitivity_by_type": {t: round(float((p[y == i] == i).mean()), 3) for i, t in enumerate(TYPES) if (y == i).any()},
           "val_best": best, "history": history, "confusion": cm.to_dict(), "smoke": smoke, "types": TYPES}
    rows = data["test_rows"]
    if "source" in rows:
        res["accuracy_by_source"] = {s: round(float((y[k] == p[k]).mean()), 3)
                                     for s in rows["source"].fillna("multicare").unique()
                                     for k in [(rows["source"].fillna("multicare") == s).to_numpy()]}
    (RUN / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("history", "confusion")}, indent=1))
    print(cm.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--add-hospital", action="store_true", help="after --prepare: add real hospital pictures (labels_v2.csv)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    prepare() if a.prepare else add_hospital() if a.add_hospital else train(a.smoke, a.epochs)
