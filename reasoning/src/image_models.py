"""The trained image models, as the app runs them (8 Nov 2026, fix plan 5o items 1, 2 and 4). All on the CPU.

    from image_models import read_picture
    out = read_picture("upload.jpg", age=67, note="fell, hit head")   # sorts it, then runs the model for its kind

1. Sorter (vision/image_type, EfficientNet-B0): bone X-ray, chest X-ray, CT, MRI, ultrasound, ECG or other.
   Test: 93.5% on 3,520 pictures; 100% on 200 each of VinDr, GRAZ, FracAtlas, RSNA head CT. Replaces MedGemma's
   guess (9/10 in the real-AI check).
2. Bone X-ray: fracture detector v3 (YOLO11s, GRAZPEDWRI-DX children's wrists + FracAtlas adults, adult X-rays
   repeated 4x). The cut-off depends on age, because one cut-off for all left adults behind:
     child (< 18): set on all validation X-rays (mostly children)   -> children's wrists AUROC 0.989
     adult or age unknown: set on adult validation X-rays (catch 90%) -> adults caught 82.5%, 77% normal unflagged
3. Chest X-ray: VinDr-CXR detector (YOLO11s, 14 findings with boxes), one cut-off per finding set on validation
   (catch 80%). Pneumonia-type shadowing = consolidation, lung opacity or infiltration (children's pneumonia test set:
   caught 85%, 82% normal unflagged, better than CheXpert's 97% / 51%). Pneumothorax has two levels (app_cutoffs.py,
   VinDr test, 15 pneumothoraces): red = the detector's box (caught 53%, 99.5% of normal X-rays unflagged); amber
   "possible, check" = only the CheXpert classifier is above its cut-off. Red or amber together catch 87%, but amber
   shows on about half of all chest X-rays, so it is a prompt to look at the lung edges, not an alarm.
4. CT slice + the note says head: bleed model (EfficientNet-B0, RSNA 2019): any bleed + 5 types, per-patient test AUROC
   0.988. Where: Grad-CAM box around what drove the score; RSNA has no drawn boxes, so it is marked "approximate".
   It was trained on CT windows packed into colour; the app uses a copy fine-tuned on grey slices too (app_cutoffs.py
   ich_grey): per-patient AUROC 0.985 on plain grey slices (original 0.904), 0.987 on colour ones.
MRI, ultrasound, ECG, other: no trained detector; MedGemma's checklist only (pipeline.py).
Every result says it is decision support with its known miss rate; boxes are where to look, not a diagnosis.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VISION = Path(__file__).resolve().parents[2] / "vision"
SORTER = VISION / "image_type" / "runs" / "image_type" / "model.pt"
FRACTURE = VISION / "runs" / "fracture_v3" / "weights" / "best.pt"
CHEST = VISION / "runs" / "chest_vindr" / "weights" / "best.pt"
ICH = VISION / "ct_head" / "runs" / "ich" / "model.pt"
ICH_GREY = VISION / "ct_head" / "runs" / "ich" / "model_grey.pt"
CUTOFFS = VISION / "results" / "app_cutoffs.json"            # from the vanda job (pneumothorax, grey CT)

TYPES = ["bone_xray", "chest_xray", "ct_slice", "mri_slice", "ultrasound", "ecg", "other"]
FRACTURE_CUT = {"child": 0.6636, "adult": 0.0137}             # vision/results/fracture_v3.json, external_tests.json
FRACTURE_NOTE = {"child": "children's wrist X-rays: catches about 8 in 10 fractures, rarely marks a normal X-ray",
                 "adult": "adult X-rays: misses about 1 in 6 fractures and marks about 1 in 4 normal X-rays"}
CHEST_NAMES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
               "Lung opacity", "Nodule/mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
               "Pulmonary fibrosis"]
CHEST_CUT = {"Aortic enlargement": 0.5459, "Atelectasis": 0.0026, "Calcification": 0.0123, "Cardiomegaly": 0.5409,
             "Consolidation": 0.0104, "ILD": 0.0475, "Infiltration": 0.0394, "Lung opacity": 0.0303, "Nodule/mass": 0.036,
             "Other lesion": 0.0101, "Pleural effusion": 0.1078, "Pleural thickening": 0.0397, "Pneumothorax": 0.0768,
             "Pulmonary fibrosis": 0.049}                          # vision/results/chest_vindr.json (catch 80% on val)
# what the ED acts on: only these are flagged red and boxed. The others (ILD, fibrosis, pleural thickening,
# calcification, "other lesion") are long-standing, and with 14 findings at 80%-catch cut-offs almost every X-ray
# would get a red flag (first laptop test: 4-6 per abnormal X-ray), so they are listed as "also seen".
CHEST_ED = ["Pneumothorax", "Pleural effusion", "Pneumonia-type shadowing", "Cardiomegaly", "Aortic enlargement",
            "Nodule/mass", "Atelectasis"]
CHEX_FILE = Path.home() / ".torchxrayvision" / "models_data" / "chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
PNEUMONIA = ("Consolidation", "Lung opacity", "Infiltration")
CHEX_PNEUMOTHORAX_CUT = 0.0683                                     # CheXpert, catch 80% on VinDr val (external_tests.json)
ICH_CUT = 0.5358                                                   # catch 90% of validation bleed slices
ICH_TYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
HEAD_WORDS = r"\b(head|brain|skull|cranial|intracranial|ct ?h|cth|ct brain|gcs|headache|stroke|seizure|confus\w*|" \
             r"collapse\w*|unconscious|loc|hit (?:his |her |their )?head|neuro\w*|subdural|bleed)\b"
_cache = {}


def _cutoffs():
    """Cut-offs re-set on validation by app_cutoffs.py, when its results are on this computer."""
    if "cut" not in _cache:
        _cache["cut"] = json.loads(CUTOFFS.read_text()) if CUTOFFS.exists() else {}
    return _cache["cut"]


def available():
    return {"sorter": SORTER.exists(), "fracture": FRACTURE.exists(), "chest": CHEST.exists(), "ich": ICH.exists()}


def _effnet(path, n):
    if path not in _cache:
        import torch
        from torchvision.models import efficientnet_b0
        m = efficientnet_b0()
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, n)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        _cache[path] = m.eval()
    return _cache[path]


def _yolo(path):
    if path not in _cache:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import YOLO
        _cache[path] = YOLO(str(path))
    return _cache[path]


def _norm():
    from torchvision import transforms as T
    return T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def sort_type(path):
    """(kind, {kind: probability}). The trained sorter; the clinician can still correct it."""
    import torch
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _norm()])
    with torch.no_grad():
        p = torch.softmax(_effnet(SORTER, len(TYPES))(tf(Image.open(path).convert("RGB"))[None]), 1)[0].numpy()
    return TYPES[int(p.argmax())], {t: round(float(v), 3) for t, v in zip(TYPES, p)}


def _draw(path, boxes):
    """The picture with labelled boxes: red = above the cut-off."""
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    w = max(2, img.width // 250)
    font = ImageFont.load_default(size=max(14, img.width // 28))
    for b in boxes:
        x0, y0, x1, y1 = b["box"]
        colour = (220, 30, 30) if b.get("flag", True) else (240, 160, 0)
        d.rectangle((x0, y0, x1, y1), outline=colour, width=w)
        label = b["name"] + (" (rough area)" if b.get("approx") else "")
        tx, ty = x0 + w, max(0, y0 - font.size - 2 * w)
        box = d.textbbox((tx, ty), label, font=font)
        d.rectangle((box[0] - w, box[1] - w, box[2] + w, box[3] + w), fill=(255, 255, 255))
        d.text((tx, ty), label, fill=colour, font=font)
    return np.asarray(img)


def _top_boxes(model, path, imgsz, names):
    """{class name: (best score, best box)} for every class with any box at conf 0.001."""
    r = model.predict(str(path), conf=0.001, imgsz=imgsz, device="cpu", verbose=False)[0]
    best = {}
    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        n = names[int(c)]
        if n not in best or s > best[n][0]:
            best[n] = (float(s), [float(v) for v in b])
    return best, r


def fracture(path, age=None):
    group = "child" if age is not None and age == age and float(age) < 18 else "adult"
    cut = FRACTURE_CUT[group]
    r = _yolo(FRACTURE).predict(str(path), conf=0.001, imgsz=640, device="cpu", verbose=False)[0]
    marks = sorted([{"name": "Fracture", "score": float(s), "box": [float(v) for v in b], "flag": True}
                    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
                    if int(c) == 0 and s >= cut], key=lambda m: -m["score"])[:3]
    top = max([float(s) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()) if int(c) == 0], default=0.0)
    who = "child" if group == "child" else ("adult" if age is not None and age == age else "adult (age not recorded)")
    line = (f"Fracture detector ({who} cut-off): {len(marks)} possible fracture(s) marked"
            if marks else f"Fracture detector ({who} cut-off): nothing marked")
    return {"model": "fracture detector v3", "findings": [{"name": "Fracture", "score": round(top, 3), "cut": cut,
                                                           "flag": bool(marks)}],
            "boxes": marks, "overlay": _draw(path, marks) if marks else None,
            "outline_box": marks[0]["box"] if marks else None, "outline_kind": "fracture",
            "line": f"{line} ({FRACTURE_NOTE[group]}).", "group": group}


def _chexpert_pneumothorax(path):
    from image_tools import CHEST_WEIGHTS, _import_xrv
    import torch
    if CHEST_WEIGHTS == "densenet121-res224-chex" and not CHEX_FILE.exists():
        raise FileNotFoundError("CheXpert weights not on this computer")       # never download them silently
    xrv = _import_xrv()
    if "xrv" not in _cache:
        _cache["xrv"] = xrv.models.DenseNet(weights=CHEST_WEIGHTS).eval()
    net = _cache["xrv"]
    img = xrv.datasets.normalize(np.asarray(Image.open(path).convert("L"), np.float32), 255)[None]
    img = xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(img))
    with torch.no_grad():
        out = net(torch.from_numpy(img)[None])[0]
    return float(out[list(net.pathologies).index("Pneumothorax")])


def chest(path):
    best, _ = _top_boxes(_yolo(CHEST), path, 1024, CHEST_NAMES)
    cuts = dict(CHEST_CUT)
    chex_cut = _cutoffs().get("pneumothorax", {}).get("chexpert_cut", CHEX_PNEUMOTHORAX_CUT)
    findings, boxes = [], []
    for n in CHEST_NAMES:
        if n in PNEUMONIA:
            continue
        s, b = best.get(n, (0.0, None))
        above = s >= cuts[n] and s > 0
        flag = above and n in CHEST_ED
        f = {"name": n, "score": round(s, 3), "cut": cuts[n], "flag": bool(flag), "also_seen": bool(above and not flag)}
        if n == "Pneumothorax":
            try:
                cx = _chexpert_pneumothorax(path)
                f["second_opinion"] = round(cx, 3)
                if cx >= chex_cut and not flag:
                    f["possible"] = True                  # amber: a prompt to check, not a red flag
            except Exception as e:                     # CheXpert weights not on this computer
                f["second_opinion_error"] = type(e).__name__
        findings.append(f)
        if flag:
            boxes.append({"name": n, "score": s, "box": b, "flag": True})
    pn = [(best.get(n, (0.0, None)), n) for n in PNEUMONIA]
    hits = [(s, b, n) for (s, b), n in pn if s > 0 and s >= cuts[n]]
    top = max((s for (s, _), _ in pn), default=0.0)
    findings.append({"name": "Pneumonia-type shadowing", "score": round(top, 3), "cut": None, "flag": bool(hits),
                     "from": "consolidation, lung opacity or infiltration"})
    if hits:
        s, b, n = max(hits)
        boxes.append({"name": n, "score": s, "box": b, "flag": True})
    order = {n: i for i, n in enumerate(CHEST_ED)}
    findings.sort(key=lambda f: (not f["flag"], not f.get("possible"), order.get(f["name"], 99)))
    flagged = [f["name"] for f in findings if f["flag"]]
    possible = [f["name"] for f in findings if f.get("possible")]
    also = [f["name"] for f in findings if f.get("also_seen")]
    second = next((f for f in findings if f["name"] == "Pneumothorax"), {})
    line = ("Chest X-ray detector: " + (", ".join(flagged) + " marked" if flagged else "no urgent finding above its cut-offs")
            + (f"; possible {', '.join(possible).lower()} (second model only: check the lung edges)" if possible else "")
            + (f"; also seen, usually not urgent: {', '.join(also)}" if also else "")
            + " (each finding misses about 1 in 5 or more; pneumothorax "
            + ("also checked by a second model)." if "second_opinion" in second else
               "checked by the detector only: the second model is not on this computer)."))
    boxes.sort(key=lambda b: (order.get(b["name"], order["Pneumonia-type shadowing"] if b["name"] in PNEUMONIA else 99)))
    main = boxes[0] if boxes else None
    return {"model": "VinDr chest detector", "findings": findings, "boxes": boxes,
            "overlay": _draw(path, boxes) if boxes else None,
            "outline_box": main["box"] if main else None, "outline_kind": "other", "line": line}


def _is_grey(img):
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 3


def _gradcam_box(model, x):
    """Box around the area that drove the 'any bleed' score (Grad-CAM on the last feature map)."""
    import torch
    feats = {}
    h = model.features[-1].register_forward_hook(lambda m, i, o: feats.update(a=o))
    x = x.clone().requires_grad_(True)
    out = model(x)
    a = feats["a"]
    grad = torch.autograd.grad(out[0, 0], a)[0]
    h.remove()
    cam = torch.relu((grad.mean((2, 3), keepdim=True) * a).sum(1))[0].detach().numpy()
    cam = cam / (cam.max() + 1e-8)
    ys, xs = np.nonzero(cam >= 0.5)
    if not len(ys):
        return None, cam
    sy, sx = 1 / cam.shape[0], 1 / cam.shape[1]
    return (xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy), cam


def head_ct(path, note=""):
    if not re.search(HEAD_WORDS, (note or "").lower()) and not re.search(r"head|brain|ct_?h", Path(path).stem.lower()):
        return {"model": "head-CT bleed model", "findings": [], "boxes": [], "overlay": None, "outline_box": None,
                "line": "CT picture: the bleed check runs only on head CT, and the notes don't say this is a head CT "
                        "(add a note such as 'CT head' to run it)."}
    import torch
    from torchvision import transforms as T
    img = Image.open(path).convert("RGB")
    grey = _is_grey(img)
    # the copy fine-tuned with half the slices grey (app_cutoffs.py ich_grey, job 1533183): per-patient test AUROC
    # 0.985 on grey slices (the original: 0.904) and 0.987 on the colour-window slices (original 0.988), so it is used
    # for every CT picture, with its cut-off set on grey validation slices
    use_grey_model = ICH_GREY.exists()
    model = _effnet(ICH_GREY if use_grey_model else ICH, 6)
    cut = _cutoffs().get("ich_grey", {}).get("grey_model_cut_off_90pct_grey_val", ICH_CUT) if use_grey_model else ICH_CUT
    x = T.Compose([T.Resize((256, 256)), T.ToTensor(), _norm()])(img)[None]
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0].numpy()
    flag = bool(p[0] >= cut)
    findings = [{"name": "Bleeding inside the skull", "score": round(float(p[0]), 3), "cut": round(cut, 4), "flag": flag}]
    findings += [{"name": t, "score": round(float(s), 3), "cut": None, "flag": False, "type_of_bleed": True}
                 for t, s in zip(ICH_TYPES, p[1:])]
    boxes, outline_box = [], None
    if flag:
        frac, _ = _gradcam_box(model, x)
        if frac:
            W, H = img.size
            box = [frac[0] * W, frac[1] * H, frac[2] * W, frac[3] * H]
            boxes = [{"name": "Bleed", "score": float(p[0]), "box": box, "flag": True, "approx": True}]
            outline_box = box
    caveat = ("trained on CT windows packed into colour; this picture is grey, so the score is less reliable"
              if grey and not use_grey_model else "")
    # the type of bleed is kept in the findings but not written: it named an epidural bleed "subdural" in the
    # group test pack (epidural type AUROC 0.87, the weakest), and a wrong type in the record could mislead
    line = ("Head-CT bleed check: possible bleed inside the skull; the box is a rough area" if flag
            else "Head-CT bleed check: no bleed above its cut-off")
    line += f" (catches about 9 in 10 bleed slices{'; ' + caveat if caveat else ''})."
    return {"model": "head-CT bleed model", "findings": findings, "boxes": boxes,
            "overlay": _draw(path, boxes) if boxes else None, "outline_box": outline_box, "outline_kind": "other",
            "line": line, "grey_input": grey}


def read_picture(path, kind=None, age=None, note=""):
    """Sort (unless the kind is given), then the trained model for that kind. Returns plain data + pictures."""
    sorted_by = None
    probs = None
    if kind is None:
        if SORTER.exists():
            kind, probs = sort_type(path)
            sorted_by = "image sorter"
        else:
            kind = None
    out = {"kind": kind, "sorted_by": sorted_by, "sorter": probs, "findings": [], "boxes": [], "overlay": None,
           "outline_box": None, "line": ""}
    if kind == "bone_xray" and FRACTURE.exists():
        out.update(fracture(path, age))
    elif kind == "chest_xray" and CHEST.exists():
        out.update(chest(path))
    elif kind == "ct_slice" and ICH.exists():
        out.update(head_ct(path, note))
    return out
