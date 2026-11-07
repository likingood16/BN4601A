#!/bin/bash
# Run ONCE on the HPC LOGIN NODE after sync_pipeline.ps1: libraries + test data for the image tools, then queues the test.
#   bash reasoning/hpc/setup_tools.sh
set -e
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
export HOME=$ROOT/home HF_HOME           # $HOME/.torchxrayvision holds the chest model weights (real home isn't writable)
mkdir -p "$HOME"
PIP="$VENV/bin/pip"
BEFORE=$($PIP freeze | grep -E '^(torch|torchvision|transformers|peft)==')
$PIP install --no-deps torchxrayvision
$PIP install -c "$ROOT/constraints.txt" scikit-image imageio
AFTER=$($PIP freeze | grep -E '^(torch|torchvision|transformers|peft)==')
[ "$BEFORE" = "$AFTER" ] && echo "torch, transformers and peft unchanged" || { echo "VERSIONS CHANGED: $AFTER"; exit 1; }
"$VENV/bin/python" -u reasoning/hpc/prepare_image_tests.py
qsub -q $GPU_QUEUE -l $GPU_SELECT -l walltime=04:00:00 -o reasoning/outputs/logs reasoning/hpc/jobs/image_tools.pbs
qstat -u "$USER"
