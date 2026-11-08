"""Cut-offs and checks for the image models as the APP uses them (8 Nov 2026, fix plan 5o items 1 and 4).

    python app_cutoffs.py --parts pneumothorax,ich_grey      # GPU job: PART=cutoffs in hpc/jobs/train_images.pbs

1. pneumothorax: the VinDr detector catches 53% of test pneumothoraces (many get no box at all). Here it is combined
   with the CheXpert classifier the app already runs: flag if EITHER is above its cut-off. Both cut-offs are chosen
   together on VinDr VALIDATION X-rays: the pair that catches >= 90% of validation pneumothoraces with the fewest
   false flags; then tested once on the test X-rays.
2. ich_grey: the bleed model trained on RSNA JPGs with 3 CT windows packed into the colour channels (brain,
   subdural, full range). A picture a clinician uploads is usually one grey brain-window slice. Test: every test
   slice turned grey (brain channel copied to all 3). If that loses accuracy, the model is fine-tuned for 1 pass with
   half the slices turned grey (model_grey.pt), the cut-off is set again on validation, and both inputs are tested.
Saves runs/app_cutoffs.json. Cut-offs never come from test pictures.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_external import CHEST, auroc, rates, top_scores, xrv_scores, _effnet, _run

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "app_cutoffs.json"
SEED = 42


def pneumothorax(res):
    from ultralytics import YOLO
    m, root = YOLO(str(HERE / "runs" / "chest_vindr" / "weights" / "best.pt")), HERE / "chest" / "data" / "vindr"
    k = CHEST.index("Pneumothorax")

    def split(name):
        imgs = sorted((root / "images" / name).glob("*"))
        y = np.array([any(l.split()[0] == str(k) for l in (root / "labels" / name / f"{p.stem}.txt").read_text().splitlines())
                      if (root / "labels" / name / f"{p.stem}.txt").exists() else False for p in imgs])
        det = top_scores(m, [str(p) for p in imgs], 1024, 14)[:, k]
        chex = xrv_scores([str(p) for p in imgs])["Pneumothorax"]
        return y, det, chex
    yv, dv, cv = split("val")
    yt, dt, ct = split("test")
    best = None
    for qd in np.linspace(0, 1, 41):                    # detector cut-off = quantile of its positive scores > 0
        pos = dv[yv & (dv > 0)]
        cd = float(np.quantile(pos, qd)) if len(pos) else 1.0
        for qc in np.linspace(0, 0.5, 51):
            cc = float(np.quantile(cv[yv], qc))
            flag = (dv >= cd) & (dv > 0) | (cv >= cc)
            sens, spec = flag[yv].mean(), (~flag[~yv]).mean()
            if sens >= 0.90 and (best is None or spec > best[2]):
                best = (cd, cc, spec, sens)
    cd, cc = best[0], best[1]
    flag_t = (dt >= cd) & (dt > 0) | (ct >= cc)
    res["pneumothorax"] = {
        "rule": "flag if the detector box score >= detector_cut OR the CheXpert score >= chexpert_cut",
        "detector_cut": round(cd, 4), "chexpert_cut": round(cc, 4),
        "val": {"n_with": int(yv.sum()), "sensitivity": round(float(best[3]), 3), "specificity": round(float(best[2]), 3)},
        "test_or_rule": rates(yt, flag_t),
        "test_detector_alone_old_cut": rates(yt, dt >= 0.0768),
        "test_auroc_detector": auroc(yt, dt), "test_auroc_chexpert": auroc(yt, ct),
        "test_auroc_max_rank": auroc(yt, np.maximum(pd.Series(dt).rank(pct=True), pd.Series(ct).rank(pct=True))),
    }


def _grey(t):
    return t[:1].expand(3, -1, -1).clone()           # brain-window channel copied to all 3 (before normalising)


def _tf(grey):
    from torchvision import transforms as T
    steps = [T.Resize((256, 256)), T.ToTensor()]
    if grey:
        steps.append(T.Lambda(_grey))
    return T.Compose(steps + [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def _per_patient(df, scores, cut):
    df = df.assign(score=scores)
    per = df.groupby("PatientID").agg(bleed=("any", "max"), score=("score", "max"))
    sl = rates(df["any"], df["score"] >= cut)
    return {"slice_auroc": auroc(df["any"], df["score"]), "slice": sl,
            "patient_auroc": auroc(per["bleed"], per["score"]), "patient": rates(per["bleed"], per["score"] >= cut)}


def _fine_tune_grey(run, tr, va):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _effnet(run / "model.pt", 6).to(dev)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    aug = T.Compose([T.Resize((256, 256)), T.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                     T.RandomHorizontalFlip(), T.ToTensor()])
    labels = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]

    class Slices(torch.utils.data.Dataset):
        def __len__(self):
            return len(tr)

        def __getitem__(self, i):
            r = tr.iloc[i]
            x = aug(Image.open(r.path).convert("RGB"))
            if torch.rand(1).item() < 0.5:
                x = _grey(x)
            return norm(x), torch.tensor(r[labels].to_numpy(float), dtype=torch.float32)
    dl = torch.utils.data.DataLoader(Slices(), batch_size=96, shuffle=True, num_workers=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    for x, y in dl:
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(model(x.to(dev)), y.to(dev)).backward()
        opt.step()
    torch.save(model.state_dict(), run / "model_grey.pt")
    return model.eval()


def ich_grey(res):
    run = HERE / "ct_head" / "runs" / "ich"
    df = pd.read_csv(run / "slices.csv")
    old_cut = json.loads((run / "test_metrics.json").read_text())["any_cut_off_90pct_val"]
    va = df[df["split"] == "val"].sample(min(20000, (df["split"] == "val").sum()), random_state=SEED)
    te = df[df["split"] == "test"].sample(min(40000, (df["split"] == "test").sum()), random_state=SEED)
    m = _effnet(run / "model.pt", 6)
    out = {"original_cut_off": old_cut}
    for grey in (False, True):
        out[f"original_model_{'grey' if grey else 'colour'}_input"] = _per_patient(te, _run(m, te["path"].tolist(), _tf(grey), True)[:, 0], old_cut)
    if out["original_model_grey_input"]["patient_auroc"] < 0.95:
        tr = df[df["split"] == "train"]
        pos = tr[tr["any"] == 1]
        tr = pd.concat([pos, tr[tr["any"] == 0].sample(len(pos), random_state=SEED)]).sample(frac=0.5, random_state=SEED)
        g = _fine_tune_grey(run, tr, va)
        sv = _run(g, va["path"].tolist(), _tf(True), True)[:, 0]
        cut = float(np.quantile(sv[va["any"].to_numpy() == 1], 0.1))      # catches 90% of grey validation bleed slices
        out["grey_model_cut_off_90pct_grey_val"] = round(cut, 4)
        for grey in (False, True):
            out[f"grey_model_{'grey' if grey else 'colour'}_input"] = _per_patient(te, _run(g, te["path"].tolist(), _tf(grey), True)[:, 0], cut)
    res["ich_grey"] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="pneumothorax,ich_grey")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for part in a.parts.split(","):
        print("==", part, flush=True)
        globals()[part](res)
        OUT.write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res[part], indent=1, default=str), flush=True)
