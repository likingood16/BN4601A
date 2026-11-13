#!/bin/bash
# Run on the HPC LOGIN NODE (has internet) after the code is uploaded: fetches what the offline GPU jobs need,
# checks it, then queues both image upgrades.   bash reasoning/hpc/start_image_upgrade.sh
set -e
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
export HOME=$ROOT/home HF_HOME
mkdir -p "$HOME"
# Ultralytics draws training plots with matplotlib, which the venv lacks (torch etc. pinned by the constraints file)
"$VENV/bin/pip" install -q -c "$ROOT/constraints.txt" matplotlib
cd vision/detector
# YOLO11m-seg (about 45 MB) + YOLO11n (Ultralytics' mixed-precision self-check) + the plot font, from Ultralytics' GitHub
"$VENV/bin/python" -c "
from ultralytics import YOLO
from ultralytics.utils.checks import check_font
YOLO('yolo11m-seg.pt'); YOLO('yolo11n.pt'); check_font('Arial.ttf'); print('ultralytics files ready')"
"$VENV/bin/python" -c "
from transformers import SamModel; SamModel.from_pretrained('wanglab/medsam-vit-base'); print('MedSAM in cache')"
ls data/yolo/splits.csv data/yolo/images/train | head -2
cd ../..
Q="-q $GPU_QUEUE -l $GPU_SELECT -o reasoning/outputs/logs"
qsub $Q -l walltime=08:00:00 -v PART=detector reasoning/hpc/jobs/image_upgrade.pbs
qsub $Q -l walltime=03:00:00 -v PART=medsam reasoning/hpc/jobs/image_upgrade.pbs
qstat -u "$USER"
