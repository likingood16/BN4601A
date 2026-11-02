#!/bin/bash
# Queue the task-adapter steps so each starts only if the previous one finished OK.
#   bash reasoning/hpc/chain_task.sh <task_data job id>
cd "$(dirname "$0")/../.."
source reasoning/hpc/hpc.conf
LOGS="$PWD/reasoning/outputs/logs"
TRAIN=$(qsub -q "$GPU_QUEUE" -l "$GPU_SELECT" -l walltime=10:00:00 -o "$LOGS" -W depend=afterok:"$1" reasoning/hpc/jobs/train_task.pbs)
TEST=$(qsub -q "$GPU_QUEUE" -l "$GPU_SELECT" -l walltime=08:00:00 -o "$LOGS" -W depend=afterok:"$TRAIN" reasoning/hpc/jobs/test_task.pbs)
echo "train_task: $TRAIN (after $1)"
echo "test_task:  $TEST (after $TRAIN)"
qstat -u "$USER"
