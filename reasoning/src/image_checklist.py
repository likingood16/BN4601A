"""Type-aware image reading (7 Nov 2026): each kind of picture is checked for what the ED looks for on it, and the
findings are located and outlined. Fractures were only one case of this.

    found = checklist(model, processor, "cxr.jpg", "chest_xray")   # {"Pneumothorax": {"seen": "no", ...}, ...}
    box = locate(model, processor, "cxr.jpg", "Pleural effusion")    # (x0, y0, x1, y1) in pixels, or None
    anatomy = chest_anatomy("cxr.jpg")                               # heart + lung outlines, cardiothoracic ratio

What runs on each kind (pipeline.read_image puts it together):
  every kind   MedGemma answers the ED checklist for that kind (yes / no / unclear + where). Plain MedGemma, not
               Cato's adapter: the adapter writes figure-caption style text (real-AI check, 7 Nov).
  "yes" items  MedGemma gives a box (MedGemma 1.5 was trained to localise findings on chest X-rays; on other kinds
               this is untested, so it's marked experimental), then MedSAM outlines inside the box.
  bone X-ray   + fracture detector -> fine-tuned MedSAM outline (Dice 0.84 on FracAtlas test)
  chest X-ray  + 18-finding chest model (pneumonia AUROC 0.90) + heart/lung outlines (ChestX-Det PSPNet) and the
               cardiothoracic ratio (heart width / chest width; above 0.5 on a PA film suggests a big heart)
Nothing here is a diagnosis: every line on the record says it's an AI check that only points where to look.
"""
import json
import re

import numpy as np
from PIL import Image

from answers import final_answer
from image_reader import image_messages
from model import ask_messages

# what an ED clinician checks on each kind of picture (the common ED questions, not a full radiology report)
CHECKLIST = {
    "bone_xray": ["Fracture", "Dislocation", "Joint effusion", "Foreign body"],
    "chest_xray": ["Pneumothorax", "Consolidation or pneumonia", "Pleural effusion", "Enlarged heart",
                   "Pulmonary oedema", "Free air under the diaphragm", "Rib fracture"],
    "ct_slice": ["Bleeding inside the skull", "Midline shift or mass effect", "Fracture", "Free air",
                 "Free fluid", "Bowel obstruction", "Aortic aneurysm or dissection", "Mass"],
    "mri_slice": ["Acute stroke", "Bleeding", "Mass", "Spinal cord compression"],
    "ultrasound": ["Free fluid", "Pericardial effusion", "Aortic aneurysm", "Gallstones or inflamed gallbladder",
                   "Deep vein thrombosis", "Pregnancy outside the womb"],
    "ecg": ["ST elevation", "ST depression", "Atrial fibrillation", "Heart block", "Fast heart rate",
            "Slow heart rate", "Peaked T waves"],
    "other": ["Wound or skin infection", "Swelling", "Foreign body"],
}
LOCATABLE = {"bone_xray", "chest_xray", "ct_slice", "mri_slice", "ultrasound"}   # an outline makes sense on these
NOTE = "AI check, not a radiologist report: it points where to look"


def _ask(model, processor, path, question, tokens):
    return final_answer(ask_messages(model, processor, image_messages(path, question), tokens, False)) or ""


def _ask_all(model, processor, path, questions, tokens):
    """All the checklist questions about one picture in ONE batched call (left-padded), so the picture and prompts are
    read side by side instead of 4-8 times one after another (one question took 23-76 s on the laptop, 9 Nov, 5q #5).
    Falls back to one question at a time if the batch doesn't fit in memory or the processor can't batch."""
    import torch
    from model import compute_dtype
    try:
        tok = processor.tokenizer
        side, tok.padding_side = tok.padding_side, "left"
        try:
            inputs = processor.apply_chat_template([image_messages(path, q) for q in questions], add_generation_prompt=True,
                                                   tokenize=True, return_dict=True, return_tensors="pt",
                                                   processor_kwargs={"padding": True})
        finally:
            tok.padding_side = side
        inputs = inputs.to(model.device, dtype=compute_dtype())
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=tokens, do_sample=False,
                                 suppress_tokens=[tok.convert_tokens_to_ids("<unused94>")])
        n = inputs["input_ids"].shape[-1]
        return [final_answer(processor.decode(o[n:], skip_special_tokens=True)) or "" for o in out]
    except Exception as e:                           # e.g. out of GPU memory on a small card
        print(f"checklist: batched call failed ({type(e).__name__}); asking one question at a time", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [_ask(model, processor, path, q, tokens) for q in questions]


def checklist(model, processor, path, kind):
    """-> {item: {"seen": "yes" | "no" | "unclear", "where": str}} for this kind's ED checklist.
    One short question per item. (Real-AI run 4, 7 Nov: asking for all items in one JSON answer came back unreadable,
    so every item was 'unclear', even a large pleural effusion the description itself named.) The questions go to
    MedGemma together, as one batch (_ask_all)."""
    from image_reader import IMAGE_TYPES
    label = IMAGE_TYPES.get(kind, ("medical image",))[0]
    items = CHECKLIST.get(kind, CHECKLIST["other"])
    questions = [f"Look at this {label}. Is there {item.lower()}? Answer with one word, yes, no or unclear, then, if "
                 "yes, a dash and where it is in a few words." for item in items]
    answers = _ask_all(model, processor, path, questions, 24)
    out = {}
    for item, raw in zip(items, answers):
        raw = raw.strip().lower()
        word = re.match(r"\W*(yes|not sure|no|unclear|cannot|can't)\b", raw)     # "not sure" before "no"
        seen = {"not sure": "unclear", "cannot": "unclear", "can't": "unclear"}.get(word.group(1), word.group(1)) if word else "unclear"
        parts = re.split(r"\s*[-–—:,]\s*", raw, maxsplit=1)
        where = parts[1].strip(" .") if seen == "yes" and len(parts) > 1 else ""
        out[item] = {"seen": seen, "where": where[:60]}
    return out


def locate(model, processor, path, finding):
    """MedGemma's box around one finding -> (x0, y0, x1, y1) in pixels, or None. Asked on a 0-1000 scale, as
    MedGemma 1.5 was trained to give boxes; boxes that are too small or cover almost all of the picture are dropped."""
    w, h = Image.open(path).size
    q = (f"Locate the {finding.lower()} in this image. Reply with its bounding box only, as "
         "[y_min, x_min, y_max, x_max] on a 0-1000 scale.")
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", _ask(model, processor, path, q, 40))[:4]]
    if len(nums) < 4 or any(n > 1000 for n in nums):
        return None
    y0, x0, y1, x1 = nums
    x0, x1, y0, y1 = sorted((x0, x1)) + sorted((y0, y1))
    area = (x1 - x0) * (y1 - y0) / 1e6
    if not 0.002 < area < 0.9:
        return None
    return (x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000)


_pspnet = {}


def chest_anatomy(path):
    """Heart and lung outlines on a chest X-ray (ChestX-Det PSPNet via TorchXRayVision) and the cardiothoracic
    ratio: widest heart / widest chest (outer edges of the two lungs), measured row by row on the masks."""
    import torch
    from image_tools import _import_xrv, overlay
    xrv = _import_xrv()
    if "m" not in _pspnet:
        _pspnet["m"] = xrv.baseline_models.chestx_det.PSPNet().eval()
    img = xrv.datasets.normalize(np.asarray(Image.open(path).convert("L"), np.float32), 255)
    h0, w0 = img.shape                                     # the model wants a square: pad (don't crop the chest)
    side = max(h0, w0)
    square = np.full((side, side), img.min(), np.float32)
    square[:h0, :w0] = img
    with torch.no_grad():
        out = torch.sigmoid(_pspnet["m"](torch.from_numpy(square)[None, None]))[0].numpy()  # [14, 512, 512]
    keep = (slice(0, round(512 * h0 / side)), slice(0, round(512 * w0 / side)))          # drop the padding again
    t = _pspnet["m"].targets
    heart = out[t.index("Heart")][keep] > 0.5
    lungs = ((out[t.index("Left Lung")] > 0.5) | (out[t.index("Right Lung")] > 0.5))[keep]

    def widest(mask):
        cols = [np.flatnonzero(r) for r in mask]
        return max(((c[-1] - c[0]) for c in cols if len(c)), default=0)
    chest = widest(lungs)
    ctr = float(widest(heart) / chest) if chest else None
    w, h = Image.open(path).size
    big = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h))) / 255.0
    picture = overlay(path, big(lungs), (60, 140, 255))
    picture = overlay_on(picture, big(heart), (255, 80, 80))
    return {"ctr": ctr, "heart_found": bool(heart.any()), "lungs_found": bool(lungs.any()), "overlay": picture}


def overlay_on(picture, mask, colour):
    """Paint a 0-1 mask onto an RGB array that already has other outlines."""
    a = mask[..., None] * 0.45
    return (picture * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)


def checklist_line(kind, found, ctr=None):
    """The text for the record: positives first, then what was checked and not seen."""
    yes = [f"{k}{' (' + v['where'] + ')' if v['where'] else ''}" for k, v in found.items() if v["seen"] == "yes"]
    unclear = [k for k, v in found.items() if v["seen"] == "unclear"]
    no = [k for k, v in found.items() if v["seen"] == "no"]
    parts = [f"Possible: {', '.join(yes)}" if yes else "Nothing flagged on the checklist"]
    if unclear:
        parts.append(f"unclear: {', '.join(unclear)}")
    if no:
        parts.append(f"not seen: {', '.join(no)}")
    if ctr:
        parts.append(f"cardiothoracic ratio {ctr:.2f}{' (above 0.5: heart may be enlarged)' if ctr > 0.5 else ''}")
    return f" ED checklist ({NOTE}): " + "; ".join(parts) + "."
