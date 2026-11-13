#!/bin/bash
# Run on the HPC LOGIN NODE after sync_pipeline.ps1: installs the detector library once, then queues
# the quick trial (2 cases per check) and the full check, which only starts if the trial finished without errors.
#   bash reasoning/hpc/start_trial.sh
set -e
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
"$VENV/bin/python" -c "import ultralytics" 2>/dev/null || bash reasoning/hpc/setup_image.sh
ls reasoning/models/lora/task/stage1/adapter/adapter_config.json
Q="-q $GPU_QUEUE -l $GPU_SELECT -o reasoning/outputs/logs"
TRIAL=$(qsub $Q -l walltime=01:00:00 reasoning/hpc/jobs/pipeline_trial.pbs)
FULL=$(qsub $Q -l walltime=12:00:00 -W depend=afterok:$TRIAL reasoning/hpc/jobs/pipeline_check.pbs)
echo "trial $TRIAL, full check $FULL (starts after the trial succeeds)"
qstat -u "$USER"
