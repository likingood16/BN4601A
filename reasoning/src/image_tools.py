"""Image tools next to the image reader (6 Nov 2026): sort the image type, check chest X-rays, outline a region.

    kind, said = sort_image(model, processor, "upload.jpg")   # MedGemma picks one of IMAGE_TYPES; no training
    found = chest_findings("cxr.jpg")      # TorchXRayVision: 18 findings + a map of where it looked
    found = outline("ct.jpg", (40, 60, 200, 220))   # MedSAM: outline of what is inside a box the clinician drew

Sorting: the file type rules out video, then MedGemma answers a multiple-choice question about the picture.
Chest X-rays: a public model trained on 8 big chest X-ray collections (Cohen et al. 2022), run as is.
Other images: nothing finds things automatically in any CT/MRI/ultrasound picture, so the clinician draws a box and
MedSAM (Ma et al. 2024) outlines what is inside it. All three are decision support; the record says so.
Tested in image_tools_check.py.
"""
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from answers import final_answer
from image_reader import image_messages
from model import ask_messages

VIDEO = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".gif"}
SORT_QUESTION = ("What kind of medical image is this? Answer with the letter only.\n"
                 "A) bone or joint X-ray\nB) chest X-ray\nC) CT scan\nD) MRI scan\nE) ultrasound\nF) ECG tracing\n"
                 "G) something else (photograph, microscope slide, endoscopy, chart)")
LETTERS = dict(zip("ABCDEFG", ["bone_xray", "chest_xray", "ct_slice", "mri_slice", "ultrasound", "ecg", "other"]))
CHEST_WEIGHTS = os.environ.get("BN4601A_CHEST_WEIGHTS", "densenet121-res224-chex")   # chex: pneumonia AUROC 0.90 vs 0.79 for "all" (image_tools_chest.json)
# MedSAM with its mask decoder fine-tuned on FracAtlas outlines (vision/detector/finetune_medsam.py, job 1527113):
# test Dice 0.84 vs 0.48 for plain MedSAM and 0.39 for the box alone. Used when it's on this computer (it's on vanda).
TUNED_MEDSAM = Path(__file__).resolve().parents[2] / "vision" / "detector" / "runs" / "medsam_fracture"
MEDSAM = os.environ.get("BN4601A_MEDSAM", str(TUNED_MEDSAM) if (TUNED_MEDSAM / "model.safetensors").exists()
                        else "wanglab/medsam-vit-base")
CHEST_NOTE = "screening model for adult chest X-rays; a high score means 'look here', not a diagnosis"
_chest, _sam = {}, {}


def sort_image(model, processor, path, think=False):
    """(image type, MedGemma's answer). Unclear answers become 'other', so the clinician can correct it."""
    if Path(path).suffix.lower() in VIDEO:
        raise ValueError("Videos are not read: please upload a still image (a screenshot of the key frame).")
    said = final_answer(ask_messages(model, processor, image_messages(path, SORT_QUESTION), 16, think))
    m = re.search(r"\b([A-G])\b", said.upper())
    return (LETTERS[m.group(1)] if m else "other"), said


def overlay(path, heat, colour=(255, 0, 0)):
    """The picture with a 0-1 map painted on top in one colour."""
    img = np.asarray(Image.open(path).convert("RGB").resize(heat.shape[::-1])).astype(np.float32)
    a = (heat[..., None] * 0.55)
    return (img * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)


def _import_xrv():
    """torchxrayvision imports its own top-level package called `model`, which clashes with our src/model.py.
    Hide ours during the import, then put it back and undo the sys.path entry it adds."""
    import sys
    ours, before = sys.modules.pop("model", None), list(sys.path)
    try:
        import torchxrayvision as xrv
    finally:
        sys.modules.pop("model", None)
        if ours is not None:
            sys.modules["model"] = ours
        sys.path[:] = before
    return xrv


def chest_findings(path, weights=CHEST_WEIGHTS, top=3):
    """Scores for 18 chest findings, the top ones, and where the model looked for the top finding."""
    xrv = _import_xrv()
    if weights not in _chest:
        _chest[weights] = xrv.models.DenseNet(weights=weights).eval()
    net = _chest[weights]
    img = xrv.datasets.normalize(np.asarray(Image.open(path).convert("L"), np.float32), 255)[None]
    img = xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(img))
    x = torch.from_numpy(img)[None].requires_grad_(True)
    out = net(x)[0]
    scores = {name: float(p) for name, p in zip(net.pathologies, out) if name}
    best = max(scores, key=scores.get)
    out[list(net.pathologies).index(best)].backward()           # which pixels moved the top score most
    sal = x.grad.abs()[0, 0].numpy()
    from scipy.ndimage import gaussian_filter                   # Pillow can't blur float ('F') images
    sal = gaussian_filter(sal / (sal.max() + 1e-8), 6)
    sal = sal / (sal.max() + 1e-8)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top]
    return {"scores": scores, "top": ranked, "overlay": overlay(path, sal)}


def outline(path, box, fracture=True):
    """MedSAM mask for the region inside box = (x0, y0, x1, y1) in picture pixels. fracture=True uses the MedSAM
    fine-tuned on fracture outlines; anything else (an effusion, a bleed, free fluid) uses plain MedSAM, which was
    trained on CT, MRI, ultrasound and X-ray structures in general."""
    from transformers import SamModel, SamProcessor
    name = MEDSAM if fracture else "wanglab/medsam-vit-base"
    if name not in _sam:
        _sam[name] = SamModel.from_pretrained(name).eval()
        _sam["processor"] = SamProcessor.from_pretrained("wanglab/medsam-vit-base")   # same pre-processing for both
    image = Image.open(path).convert("RGB")
    proc = _sam["processor"]
    inputs = proc(image, input_boxes=[[list(map(float, box))]], return_tensors="pt")
    with torch.no_grad():
        out = _sam[name](**inputs, multimask_output=False)
    mask = proc.image_processor.post_process_masks(out.pred_masks, inputs["original_sizes"],
                                                   inputs["reshaped_input_sizes"])[0][0, 0].numpy()
    return {"mask": mask, "area_pct": float(mask.mean() * 100), "overlay": overlay(path, mask.astype(np.float32), (0, 160, 255))}


def tools_line(kind, found):
    """Text for the record."""
    if kind == "chest_xray":
        tops = ", ".join(f"{n} {p:.0%}" for n, p in found["top"])
        return f" Chest X-ray check: highest scores {tops} ({CHEST_NOTE})."
    return ""
