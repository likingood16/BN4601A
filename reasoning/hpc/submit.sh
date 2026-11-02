#!/bin/bash
# Submit jobs using the settings in hpc.conf. Run ON THE HPC from anywhere:
#   bash reasoning/hpc/submit.sh probe                 # 10-min GPU test
#   bash reasoning/hpc/submit.sh lora mixed            # continue the Colab run (starts from its 22,448 checkpoint)
#   bash reasoning/hpc/submit.sh lora emergency
#   bash reasoning/hpc/submit.sh lora two_stage        # 30k general, then 30k emergency
#   bash reasoning/hpc/submit.sh embed                 # embeddings + heads + base-vs-LoRA evaluation
#   bash reasoning/hpc/submit.sh all                   # the 3 LoRA arms + embed, all at once
cd "$(dirname "$0")/../.."                              # project folder (BN4601A)
source reasoning/hpc/hpc.conf
P=${PBS_PROJECT:+-P $PBS_PROJECT}
J=reasoning/hpc/jobs
LOGS="$PWD/reasoning/outputs/logs"
mkdir -p "$LOGS"

lora() {   # $1 = arm
  local vars="ARM=$1"
  if [ "$1" = mixed ] && [ ! -f reasoning/models/lora/mixed/run/progress.json ] \
     && [ -d reasoning/models/lora/mixed/stage1_22448 ]; then
    vars="$vars,INIT_FROM=reasoning/models/lora/mixed/stage1_22448"   # don't redo what Colab already trained
  fi
  [ "$1" = two_stage ] && vars="$vars,TWO_STAGE_SIZE=30000"
  qsub $P -q "$GPU_QUEUE" -l "$GPU_SELECT" -l walltime="$MAX_WALLTIME" -o "$LOGS" -N "lora_$1" -v "$vars" "$J/train_lora.pbs"
}
embed() {
  qsub $P -q "$GPU_QUEUE" -l "$GPU_SELECT" -l walltime=12:00:00 -o "$LOGS" -N embed_eval -v "ADAPTERS=$ADAPTERS" "$J/embed_heads_eval.pbs"
}

case "$1" in
  probe) qsub $P -q "$GPU_QUEUE" -l "$GPU_SELECT" -o "$LOGS" "$J/probe_gpu.pbs" ;;
  lora)  lora "$2" ;;
  embed) embed ;;
  all)   for arm in mixed emergency two_stage; do lora $arm; done; embed ;;
  *)     echo "usage: submit.sh probe | lora <mixed|emergency|two_stage> | embed | all" ;;
esac
qstat -u "$USER" 2>/dev/null
