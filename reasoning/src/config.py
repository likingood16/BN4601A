"""One place for paths, model names and seeds. Works on the laptop, Colab and NUS HPC.

Paths are found relative to this file, so code runs from any working directory.
BN4601A_DATA / BN4601A_MODELS / BN4601A_OUTPUTS can move big files elsewhere (not needed on the NUS HPC:
the whole project already sits on /scratch there).
"""
import os
import sys
from pathlib import Path

REASONING = Path(__file__).resolve().parents[1]          # .../BN4601A/reasoning
REPO = REASONING.parent                                    # .../BN4601A

DATA = Path(os.environ.get("BN4601A_DATA", REASONING / "data"))
PROCESSED = DATA / "processed"                             # made by notebook 02
MODELS = Path(os.environ.get("BN4601A_MODELS", REASONING / "models"))
LORA_DIR = MODELS / "lora"                                 # LoRA adapters: lora/<arm>/<snapshot>/adapter
HEADS_DIR = MODELS / "heads"                               # trained score heads (joblib)
OUTPUTS = Path(os.environ.get("BN4601A_OUTPUTS", REASONING / "outputs"))   # embeddings, metrics, reports

# Used for inference and embeddings. The HPC sets BN4601A_BASE_MODEL=unsloth/medgemma-1.5-4b-it: the same
# weights (identical files) but not gated, so no Hugging Face login is needed there.
BASE_MODEL = os.environ.get("BN4601A_BASE_MODEL", "google/medgemma-1.5-4b-it")
LORA_BASE_MODEL = "unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit"   # same weights, used by Unsloth for training
SEED = 42
URGENT_MAX_LEVEL = 2                                       # triage 1-2 = urgent (same as Caleb)

# Caleb's code (VitalsModel, ktas_data, news2) importable from anywhere
for p in (REPO / "vitals" / "src", REPO / "multicare_filter" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
