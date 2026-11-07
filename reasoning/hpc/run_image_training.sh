#!/bin/bash
# Run on the HPC LOGIN NODE in the background (7 Nov 2026):
#   nohup bash reasoning/hpc/run_image_training.sh > reasoning/outputs/logs/run_image_training.log 2>&1 &
# Waits for each download, prepares its dataset (this also checks the files read correctly), then queues its training.
# Model files the offline GPU nodes need are fetched first.
set -o pipefail
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
export HOME=$ROOT/home HF_HOME
LOGS=reasoning/outputs/logs
Q="-q $GPU_QUEUE -l $GPU_SELECT -o $LOGS"
say() { echo "$(date +%H:%M) $*"; }

cd vision
"$VENV/bin/python" -c "
from ultralytics import YOLO; YOLO('yolo11s.pt')
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
print('yolo11s + efficientnet_b0 weights ready')"
cd ..

queue() {  # name part walltime
  say "queue $1"; qsub $Q -l walltime=$3 -v PART=$2 reasoning/hpc/jobs/train_images.pbs
}

chest_done=0; ich_done=0; frac_done=0; type_done=0
while [ $((chest_done + ich_done + frac_done + type_done)) -lt 4 ]; do
  if [ $chest_done = 0 ] && grep -q "VinBigData PNGs" $LOGS/download_kaggle.log 2>/dev/null; then
    say "prepare chest"; (cd vision/chest && "$VENV/bin/python" -u prepare_vindr.py) && queue chest chest 16:00:00; chest_done=1
  fi
  if [ $ich_done = 0 ] && grep -q "RSNA head-CT JPGs" $LOGS/download_kaggle.log 2>/dev/null; then
    say "prepare ich"; (cd vision/ct_head && "$VENV/bin/python" -u train_ich.py --prepare) && queue ich ich 16:00:00; ich_done=1
  fi
  if [ $frac_done = 0 ] && grep -q "unzipping images_part4" $LOGS/download_image_data.log 2>/dev/null \
     && grep -q "MultiCaRe images" $LOGS/download_image_data.log 2>/dev/null; then
    say "prepare fracture"; (cd vision/detector && "$VENV/bin/python" -u prepare_combined.py) && queue fracture fracture 16:00:00; frac_done=1
  fi
  if [ $type_done = 0 ] && grep -q "all open downloads done" $LOGS/download_image_data.log 2>/dev/null; then
    say "prepare imagetype"; (cd vision/image_type && "$VENV/bin/python" -u train_image_type.py --prepare) && queue imagetype imagetype 08:00:00; type_done=1
  fi
  sleep 60
done
say "all four queued"; qstat -u "$USER"
