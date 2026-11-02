"""Whatever image the ED gets: the group's adapted model (Cato's ED-VLM) describes it; on bone X-rays the fracture
detector also draws outlines.

    model, processor = load_image_model()   # MedGemma v1 + Cato's LoRA. adapted=False gives the same base without it
    text = describe_image(model, processor, "xray.jpg", "bone_xray")
    found = detect_fracture("xray.jpg")     # {"n": 1, "top_conf": 0.29, "overlay": <RGB array with outlines>}

Only the FINDINGS step of Cato's model is used. His second step ("impression + next steps") copies the style of case
reports and invents patient details (see vision/cato/hpc_run_750/test_predictions.csv), which must not reach a record.
His findings are short and caption-like. His exact prompts aren't in the files he shared, so PROMPT is our wording.

The clinician picks the image type, so no sorter model is needed. It reads single pictures only: a whole CT scan
can't go in, only one slice. The detector runs on the CPU, so it never competes for GPU memory. This model has a
different base from the reasoning model (v1, not 1.5), so the app holds both: too much for a 4 GB laptop.
"""
import os
from pathlib import Path

from PIL import Image

from answers import final_answer
from config import REPO
from model import ask_messages, load

# image type -> (name told to MedGemma, section of the visit record)
IMAGE_TYPES = {"bone_xray": ("bone X-ray", "imaging"), "chest_xray": ("chest X-ray", "imaging"),
               "ct_slice": ("single CT slice", "imaging"), "ultrasound": ("ultrasound image", "imaging"),
               "mri_slice": ("single MRI slice", "imaging"), "ecg": ("12-lead ECG", "ecg"),
               "other": ("medical image", "imaging")}
IMAGE_BASE = os.environ.get("BN4601A_IMAGE_BASE", "google/medgemma-4b-it")     # the HPC uses the ungated unsloth copy
IMAGE_ADAPTER = REPO / "vision" / "cato" / "hpc_run_750" / "huggingface_adapter"
PROMPT = "Describe the visible findings on this {label}."
DETECTOR = Path(os.environ.get("BN4601A_DETECTOR", REPO / "vision" / "detector" / "runs" / "fracture" / "weights" / "best.pt"))
DETECTOR_CONF = 0.25
DETECTOR_NOTE = "misses about half of fractures on test X-rays, so a negative does not rule one out"
_detector = None


def load_image_model(adapted=True):
    model, processor = load(mode="4bit", model_id=IMAGE_BASE)
    if adapted:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(IMAGE_ADAPTER)).eval()
    return model, processor


def image_messages(path, question, system=None):
    image = Image.open(path).convert("RGB")
    head = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
    return head + [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]


def describe_image(model, processor, path, image_type, max_new_tokens=128):
    """128 tokens = the budget Cato trained and tested findings with (run_config.json)."""
    question = PROMPT.format(label=IMAGE_TYPES[image_type][0])
    return " ".join(final_answer(ask_messages(model, processor, image_messages(path, question), max_new_tokens)).split())


def detect_fracture(path):
    """Number of possible fractures on a bone X-ray, the top confidence, and the picture with outlines drawn on."""
    global _detector
    if _detector is None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # Windows + conda, see vision/detector/train_detector.py
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import YOLO
        _detector = YOLO(str(DETECTOR))
    result = _detector.predict(str(path), conf=DETECTOR_CONF, imgsz=640, device="cpu", verbose=False)[0]
    confs = result.boxes.conf.tolist()
    return {"n": len(confs), "top_conf": max(confs, default=0.0), "overlay": result.plot()[..., ::-1]}


def imaging_line(image_type, text, found=None):
    """The text that goes on the patient's record."""
    label = IMAGE_TYPES[image_type][0]
    line = f"{label}, AI description, not a radiologist report: {text or 'no description produced'}"
    if found is not None:
        seen = (f"{found['n']} possible fracture(s) marked, top confidence {found['top_conf']:.0%}" if found["n"]
                else "nothing marked")
        line += f" Fracture detector: {seen} ({DETECTOR_NOTE})."
    return line
