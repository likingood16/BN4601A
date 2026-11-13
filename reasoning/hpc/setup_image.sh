#!/bin/bash
# Run ONCE on the HPC LOGIN NODE (it has internet; GPU nodes don't): what the image step needs.
#   bash reasoning/hpc/setup_image.sh
# The detector's library, without letting it change torch or transformers. Cato's model now uses MedGemma 1.5,
# which is already in the HPC cache, so no extra download.
set -e
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
export HF_HOME
PIP="$VENV/bin/pip"
BEFORE=$($PIP freeze | grep -E '^(torch|torchvision|transformers|peft)==')
# --no-deps + the headless OpenCV: the normal one needs screen libraries that GPU nodes may not have
$PIP install --no-deps ultralytics==8.4.153
$PIP install -c "$ROOT/constraints.txt" ultralytics-thop opencv-python-headless polars nvidia-ml-py
AFTER=$($PIP freeze | grep -E '^(torch|torchvision|transformers|peft)==')
[ "$BEFORE" = "$AFTER" ] && echo "torch, transformers and peft unchanged" || { echo "VERSIONS CHANGED: $AFTER"; exit 1; }
YOLO_OFFLINE=true "$VENV/bin/python" -c "from ultralytics import YOLO; m = YOLO('vision/detector/runs/fracture/weights/best.pt'); print('detector loads OK:', m.task)"
echo "setup done"
