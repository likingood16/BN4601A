"""Copy one held-out example picture of each kind into data/image_tests/demo/, for testing the app by hand.
Run on the HPC login node (app_on_vanda.ps1 does it). Sources: the MultiCaRe test pictures and the chest X-ray
pneumonia test set saved by prepare_image_tests.py, plus a FracAtlas test X-ray with a fracture."""
import io
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "reasoning" / "data" / "image_tests"
OUT = TESTS / "demo"
OUT.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv(TESTS / "multicare" / "labels.csv")
for kind, rows in labels.groupby("true_type"):
    src = TESTS / "multicare" / f"{rows.iloc[0]['file_id']}.jpg"
    shutil.copy(src, OUT / f"{kind}_multicare.jpg")

chest = pd.read_parquet(TESTS / "pneumonia_test.parquet")
label_col = next(c for c in chest.columns if c != "image")
for value in sorted(chest[label_col].unique()):
    img = chest[chest[label_col] == value].iloc[0]["image"]
    data = img["bytes"] if isinstance(img, dict) else img
    Image.open(io.BytesIO(data)).convert("RGB").save(OUT / f"chest_xray_label{value}.jpg")

split = pd.read_csv(ROOT / "vision" / "detector" / "data" / "yolo" / "splits.csv")
name = split[(split["split"] == "test") & split["fractured"]]["image_id"].iloc[0]
shutil.copy(ROOT / "vision" / "detector" / "data" / "yolo" / "images" / "test" / name, OUT / "bone_xray_fracture_fracatlas.jpg")
print(sorted(p.name for p in OUT.iterdir()))
