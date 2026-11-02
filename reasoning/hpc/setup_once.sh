#!/bin/bash
# Run ONCE on the HPC LOGIN NODE (it has internet; GPU nodes don't).
#   bash reasoning/hpc/setup_once.sh
# 1. makes a Python environment in $VENV   2. installs the libraries   3. downloads the models into $HF_HOME
set -e
cd "$(dirname "$0")/../.."                    # project folder (BN4601A)
source reasoning/hpc/hpc.conf
mkdir -p "$ROOT" "$HF_HOME"

# --- Python 3.11 (the system one is 3.9): conda from the HPC's Anaconda module, else a private Miniforge ---
PY=$(command -v python3.11 || true)
if [ -z "$PY" ] && module load Anaconda3/2023.09-0 2>/dev/null; then
  [ -x "$ROOT/py311/bin/python" ] || conda create -y -p "$ROOT/py311" python=3.11
  PY="$ROOT/py311/bin/python"
fi
if [ ! -x "$PY" ]; then
  if [ ! -x "$ROOT/miniforge3/bin/python" ]; then
    curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$ROOT/miniforge3"
  fi
  "$ROOT/miniforge3/bin/conda" create -y -p "$ROOT/py311" python=3.11
  PY="$ROOT/py311/bin/python"
fi
echo "using $PY ($($PY --version))"

# --- libraries: PyTorch for CUDA 12.6 FIRST, then lock it so nothing else can swap it ---
# (the newest default PyTorch is built for CUDA 13; this HPC's GPU driver supports CUDA 12.4)
if [ "$FRESH" = 1 ]; then rm -rf "$VENV"; fi            # FRESH=1 bash setup_once.sh  -> rebuild from scratch
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
$PIP install --upgrade pip
if ! $PIP show torch 2>/dev/null | grep -q "Version: $TORCH_VERSION+$CUDA_TAG"; then   # skip if already there
  $PIP install "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"
fi
$PIP freeze | grep -E '^(torch|torchvision)==' > "$ROOT/constraints.txt"
echo "locked:"; cat "$ROOT/constraints.txt"
# Unsloth ALONE first, so it chooses its own matching transformers / trl / peft (as on Colab);
# listing those next to it made pip fall back to an old 2025 Unsloth.
$PIP install -c "$ROOT/constraints.txt" --extra-index-url "$TORCH_INDEX" unsloth
$PIP install -c "$ROOT/constraints.txt" --extra-index-url "$TORCH_INDEX" \
    datasets bitsandbytes pandas pyarrow scipy scikit-learn==1.9.1
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
for p in ["torch", "torchvision", "transformers", "unsloth", "unsloth_zoo", "trl", "peft", "bitsandbytes", "scikit-learn"]:
    try: print(f"  {p:14s} {m.version(p)}")
    except m.PackageNotFoundError: print(f"  {p:14s} MISSING")
PY
# The login node has no GPU: check torch loads (CUDA is tested by the GPU probe job)
"$VENV/bin/python" -c "import torch; print('torch loads OK:', torch.__version__, '| built for CUDA', torch.version.cuda)"

# --- models, downloaded now because jobs run offline ---
export HF_HOME
"$VENV/bin/hf" download unsloth/medgemma-1.5-4b-it                      # inference + embeddings (same weights as Google's, not gated)
"$VENV/bin/hf" download unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit      # LoRA training base
echo "setup done"
