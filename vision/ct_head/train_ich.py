"""Head-CT bleed check, one slice at a time: any intracranial haemorrhage + its 5 types (epidural, intraparenchymal,
intraventricular, subarachnoid, subdural). Head CT is the 2nd most common ED scan (10% of US ED visits, NHAMCS 2022)
and bleeding is the most common acute finding on it (about a third of positives).

    python train_ich.py --prepare     # login node: labels + patient split (no copying)
    python train_ich.py               # GPU: train, pick the epoch on validation, test once
    python train_ich.py --smoke       # 1 short epoch

Data: RSNA 2019 Intracranial Haemorrhage challenge, as JPG slices with the challenge labels
(jhoward/rsna-hemorrhage-jpg on Kaggle; non-commercial, stays on vanda). Split 70/15/15 BY PATIENT, seed 42.
Training uses every bleed slice plus as many normal slices (balanced); validation and test keep the real mix, capped
at 40,000 slices each, so AUROC and sensitivity reflect a real scan.
Model: EfficientNet-B0 (ImageNet), 6 outputs (any + 5 types), 256 px.
Where: the app draws a box around the area that drove the "any bleed" score (Grad-CAM) and MedSAM outlines inside it.
RSNA has no drawn boxes, so that location is NOT validated here and is labelled approximate in the app.
Saves runs/ich/{model.pt, test_metrics.json}.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = Path("/scratch/e0959158/bn4601a/image_data/rsna_ich")
RUN = HERE / "runs" / "ich"
LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
SEED = 42


def prepare():
    meta = pd.read_feather(next(SRC.rglob("df_trn.fth")))
    labels = pd.read_feather(next(SRC.rglob("labels.fth")))
    print("meta columns:", list(meta.columns)[:25], "\nlabel columns:", list(labels.columns), flush=True)
    key = next(c for c in ("SOPInstanceUID", "ID") if c in meta.columns)
    lkey = next(c for c in ("SOPInstanceUID", "ID") if c in labels.columns)
    df = meta[[key, "PatientID"]].merge(labels.rename(columns={lkey: key}), on=key)
    jpgs = {p.stem: p for p in SRC.rglob("*.jpg")}
    df["path"] = df[key].map(lambda k: str(jpgs.get(k, jpgs.get(f"ID_{k}", ""))) )
    df = df[df["path"] != ""]
    patients = df["PatientID"].drop_duplicates().sample(frac=1, random_state=SEED).tolist()
    n = len(patients)
    split_of = {p: "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test" for i, p in enumerate(patients)}
    df["split"] = df["PatientID"].map(split_of)
    RUN.mkdir(parents=True, exist_ok=True)
    df[["path", "PatientID", "split", *LABELS]].to_csv(RUN / "slices.csv", index=False)
    print(df.groupby("split")[LABELS].mean().round(3).assign(n=df.groupby("split").size()).to_string(), flush=True)


def sets(smoke):
    df = pd.read_csv(RUN / "slices.csv")
    tr = df[df["split"] == "train"]
    pos = tr[tr["any"] == 1]
    tr = pd.concat([pos, tr[tr["any"] == 0].sample(len(pos), random_state=SEED)]).sample(frac=1, random_state=SEED)
    va = df[df["split"] == "val"].sample(min(40000, (df["split"] == "val").sum()), random_state=SEED)
    te = df[df["split"] == "test"].sample(min(40000, (df["split"] == "test").sum()), random_state=SEED)
    if smoke:
        tr, va, te = tr.head(2000), va.head(1000), te.head(1000)
    return tr, va, te


def loader(rows, train, batch=96):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = (T.Compose([T.Resize((256, 256)), T.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                     T.RandomHorizontalFlip(), T.ToTensor(), norm]) if train
          else T.Compose([T.Resize((256, 256)), T.ToTensor(), norm]))

    class Slices(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows.iloc[i]
            return tf(Image.open(r.path).convert("RGB")), torch.tensor(r[LABELS].to_numpy(float), dtype=torch.float32)
    return torch.utils.data.DataLoader(Slices(), batch_size=batch, shuffle=train, num_workers=8)


def scores(model, dl, device):
    import torch
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(torch.sigmoid(model(x.to(device))).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train(smoke, epochs):
    import torch
    from sklearn.metrics import roc_auc_score
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = sets(smoke)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(LABELS))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl_tr, dl_va = loader(tr, True), loader(va, False)
    best, history = -1, []
    for epoch in range(1, (1 if smoke else epochs) + 1):
        model.train()
        for x, y in dl_tr:
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        y, p = scores(model, dl_va, device)
        auc = float(roc_auc_score(y[:, 0], p[:, 0]))
        history.append({"epoch": epoch, "val_auroc_any": round(auc, 4)})
        print(history[-1], flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), RUN / "model.pt")
    model.load_state_dict(torch.load(RUN / "model.pt", map_location=device))
    yv, pv = scores(model, dl_va, device)
    cut = float(np.quantile(pv[yv[:, 0] == 1, 0], 0.1))           # catches 90% of bleed slices in validation
    y, p = scores(model, loader(te, False), device)
    flag = p[:, 0] >= cut
    res = {"test_slices": int(len(y)), "test_bleed_slices": int(y[:, 0].sum()),
           "auroc": {l: round(float(roc_auc_score(y[:, i], p[:, i])), 4) for i, l in enumerate(LABELS) if y[:, i].any()},
           "any_cut_off_90pct_val": round(cut, 4), "any_sensitivity": round(float(flag[y[:, 0] == 1].mean()), 4),
           "any_specificity": round(float((~flag[y[:, 0] == 0]).mean()), 4), "history": history, "smoke": smoke,
           "train_slices": int(len(tr)), "labels": LABELS}
    (RUN / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    prepare() if a.prepare else train(a.smoke, a.epochs)
