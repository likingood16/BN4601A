"""Run ONCE on the HPC LOGIN NODE (internet): test pictures and model files for image_tools_check.py.
    HOME=$ROOT/home HF_HOME=$ROOT/hf_cache $VENV/bin/python reasoning/hpc/prepare_image_tests.py

1. MultiCaRe (the dataset we already use): streamed until 40 pictures per image type, labelled by MultiCaRe's own
   image_subtype / radiology_region. Only the rows read are downloaded (a few hundred MB), not the 3.2 GB.
2. Pneumonia test set (hf-vision/chest-xray-pneumonia, test split, 78 MB): 624 children's chest X-rays.
3. Model files: TorchXRayVision weights (2 x ~30 MB) and MedSAM (wanglab/medsam-vit-base, ~375 MB).
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

OUT = Path(__file__).resolve().parents[1] / "data" / "image_tests"
PER_TYPE = 40
OTHER = {"pathology", "medical_photograph", "endoscopy", "chart", "ophthalmic_imaging"}


def our_type(r):
    sub, region = r["image_subtype"], r["radiology_region"]
    if sub == "x_ray":
        return {"thorax": "chest_xray", "upper_limb": "bone_xray", "lower_limb": "bone_xray", "pelvis": "bone_xray"}.get(region)
    if sub in ("ct", "mri", "ultrasound"):
        return {"ct": "ct_slice", "mri": "mri_slice", "ultrasound": "ultrasound"}[sub]
    if sub == "ekg":
        return "ecg"
    return "other" if r["image_type"] in OTHER else None


(OUT / "multicare").mkdir(parents=True, exist_ok=True)
counts, rows = {}, []
for r in load_dataset("OpenMed/multicare-images", split="train", streaming=True):
    kind = our_type(r)
    if kind is None or counts.get(kind, 0) >= PER_TYPE:
        continue
    r["image"].convert("RGB").save(OUT / "multicare" / f"{r['file_id']}.jpg")
    rows.append({"file_id": r["file_id"], "true_type": kind, "image_type": r["image_type"], "image_subtype": r["image_subtype"],
                 "radiology_region": r["radiology_region"], "caption": r["caption"]})
    counts[kind] = counts.get(kind, 0) + 1
    if len(counts) == 7 and min(counts.values()) >= PER_TYPE:
        break
pd.DataFrame(rows).to_csv(OUT / "multicare" / "labels.csv", index=False)
print("MultiCaRe test pictures:", counts, flush=True)

src = hf_hub_download("hf-vision/chest-xray-pneumonia", "data/test-00000-of-00001.parquet", repo_type="dataset")
pd.read_parquet(src).to_parquet(OUT / "pneumonia_test.parquet")
print("pneumonia test set saved", flush=True)

import torchxrayvision as xrv  # noqa: E402  (downloads the weights into $HOME/.torchxrayvision)
for w in ("densenet121-res224-all", "densenet121-res224-chex"):
    xrv.models.DenseNet(weights=w)
snapshot_download("wanglab/medsam-vit-base")
print("model files ready")
