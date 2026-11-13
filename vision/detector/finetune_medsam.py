"""Fine-tune MedSAM's mask decoder on FracAtlas fracture outlines, so the automatic outline fits fractures better.

    python finetune_medsam.py --smoke          # 3 images, 1 epoch, CPU: check everything runs
    python finetune_medsam.py                  # GPU: train on train, pick the epoch on val, test once on test

Why: plain MedSAM overlapped the doctors' outlines by a median Dice of 0.48 on the FracAtlas test fractures, while
just shading the box gave 0.39 (reasoning/outputs/metrics/image_tools_outline.json). It was never trained on fractures.
What is trained: only the mask decoder (about 4M weights). The image encoder (the big part) stays frozen, so each
picture is encoded once and reused every epoch, which makes this quick.
Same test as before: box = the doctors' outline + 10% margin + 5 px, Dice on the full-size picture, the same
69 test outlines. Training uses random box jitter, so it doesn't rely on a perfect box.
Saves runs/medsam_fracture/ (load with SamModel.from_pretrained) and its test_metrics.json.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "yolo"
OUT = HERE / "runs" / "medsam_fracture"
BASE = "wanglab/medsam-vit-base"
SEED = 42


def polygons(label_file, w, h):
    """YOLO segmentation label -> list of [(x, y), ...] in pixels."""
    out = []
    for line in Path(label_file).read_text().splitlines():
        v = [float(x) for x in line.split()[1:]]
        if len(v) >= 6:
            out.append([(v[i] * w, v[i + 1] * h) for i in range(0, len(v) - 1, 2)])
    return out


def box_for(poly, w, h, jitter=0.0):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    mx, my = 0.1 * bw + 5, 0.1 * bh + 5
    j = lambda size: random.uniform(-jitter, jitter) * size
    return (max(0, min(xs) - mx + j(bw)), max(0, min(ys) - my + j(bh)),
            min(w, max(xs) + mx + j(bw)), min(h, max(ys) + my + j(bh)))


def items(split, limit=None):
    """[(path, w, h, [polygons])] for the fractured pictures of one split."""
    s = pd.read_csv(DATA / "splits.csv")
    names = s[(s["split"] == split) & s["fractured"]]["image_id"].tolist()[:limit]
    out = []
    for name in names:
        path = DATA / "images" / split / name
        w, h = Image.open(path).size
        polys = polygons(DATA / "labels" / split / f"{Path(name).stem}.txt", w, h)
        if polys:
            out.append((path, w, h, polys))
    return out


def truth_mask(poly, w, h):
    m = Image.new("L", (w, h))
    ImageDraw.Draw(m).polygon(poly, fill=1)
    return np.asarray(m, bool)


def low_res_target(mask, w, h, side=1024, low=256):
    """The doctors' outline in the decoder's 256x256 frame: resized so the long side is 1024, padded, then /4."""
    s = side / max(w, h)
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((round(w * s), round(h * s)), Image.NEAREST)
    canvas = Image.new("L", (side, side))
    canvas.paste(m, (0, 0))
    return torch.from_numpy(np.asarray(canvas.resize((low, low), Image.BILINEAR), np.float32) / 255.0)


@torch.no_grad()
def encode(model, proc, data, device):
    """Image embeddings once per picture (the encoder is frozen)."""
    out = []
    for path, w, h, polys in data:
        inputs = proc(Image.open(path).convert("RGB"), return_tensors="pt")
        emb = model.get_image_embeddings(inputs["pixel_values"].to(device))
        out.append({"emb": emb.half().cpu(), "w": w, "h": h, "polys": polys, "path": path,
                    "original_sizes": inputs["original_sizes"], "reshaped_input_sizes": inputs["reshaped_input_sizes"]})
    return out


def predict(model, proc, item, box, device):
    inputs = proc(Image.new("RGB", (item["w"], item["h"])), input_boxes=[[list(map(float, box))]], return_tensors="pt")
    out = model(image_embeddings=item["emb"].float().to(device), input_boxes=inputs["input_boxes"].to(device),
                multimask_output=False)
    return out.pred_masks[0, 0]                                    # (1, 256, 256) logits


@torch.no_grad()
def evaluate(model, proc, encoded, device):
    """Median Dice on full-size pictures, same boxes as image_tools_check.py."""
    model.eval()
    rows = []
    for it in encoded:
        for poly in it["polys"]:
            box = box_for(poly, it["w"], it["h"])
            logits = predict(model, proc, it, box, device)
            mask = proc.image_processor.post_process_masks(logits[None, None].cpu(), it["original_sizes"],
                                                           it["reshaped_input_sizes"])[0][0, 0].numpy().astype(bool)
            truth = truth_mask(poly, it["w"], it["h"])
            rows.append({"image": it["path"].name, "dice": float(2 * (mask & truth).sum() / (mask.sum() + truth.sum() + 1e-8))})
    return float(np.median([r["dice"] for r in rows])), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    from transformers import SamModel, SamProcessor
    device = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    model = SamModel.from_pretrained(BASE).to(device)
    proc = SamProcessor.from_pretrained(BASE)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("mask_decoder")
    limit = 3 if a.smoke else None
    enc = {s: encode(model, proc, items(s, limit), device) for s in ["train", "val", "test"]}
    print({s: (len(v), sum(len(i["polys"]) for i in v)) for s, v in enc.items()}, "pictures, outlines", flush=True)

    base_val, _ = evaluate(model, proc, enc["val"], device)
    base_test, base_rows = evaluate(model, proc, enc["test"], device)
    print(f"before training: val Dice {base_val:.3f}, test Dice {base_test:.3f}", flush=True)

    targets = [(i, k, low_res_target(truth_mask(p, it["w"], it["h"]), it["w"], it["h"]))
               for i, it in enumerate(enc["train"]) for k, p in enumerate(it["polys"])]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    best, best_epoch, wait, history = base_val, 0, 0, []
    OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, (1 if a.smoke else a.epochs) + 1):
        model.train()
        random.shuffle(targets)
        losses = []
        for start in range(0, len(targets), a.batch):
            opt.zero_grad()
            loss = 0
            for i, k, target in targets[start:start + a.batch]:
                it = enc["train"][i]
                logits = predict(model, proc, it, box_for(it["polys"][k], it["w"], it["h"], jitter=0.1), device)[0]
                t = target.to(device)
                prob = torch.sigmoid(logits)
                dice = 1 - (2 * (prob * t).sum() + 1) / (prob.sum() + t.sum() + 1)
                loss = loss + F.binary_cross_entropy_with_logits(logits, t) + dice
            loss = loss / len(targets[start:start + a.batch])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val, _ = evaluate(model, proc, enc["val"], device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_dice": val})
        print(history[-1], flush=True)
        if val > best:
            best, best_epoch, wait = val, epoch, 0
            model.save_pretrained(OUT)
            proc.save_pretrained(OUT)
        else:
            wait += 1
            if wait >= a.patience:
                break
    if best_epoch == 0:                       # training never beat plain MedSAM on validation: keep plain MedSAM
        print("no epoch beat plain MedSAM on validation; nothing saved", flush=True)
        test, rows = base_test, base_rows
    else:
        model = SamModel.from_pretrained(OUT).to(device)
        test, rows = evaluate(model, proc, enc["test"], device)
    res = {"base": BASE, "best_epoch": best_epoch, "val_dice_before": base_val, "val_dice_best": best,
           "test_dice_before": base_test, "test_dice_after": test, "n_test_outlines": len(rows),
           "box_only_test_dice (image_tools_outline.json)": 0.39, "history": history, "cases": rows,
           "smoke": a.smoke}
    (OUT / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "history")}, indent=1))


if __name__ == "__main__":
    main()
