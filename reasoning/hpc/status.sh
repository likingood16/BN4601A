#!/bin/bash
# Quick progress report (read-only). Run on the HPC:  bash reasoning/hpc/status.sh
cd "$(dirname "$0")/../.."
date
qstat -u "$USER"
for a in mixed emergency two_stage; do
  f=reasoning/models/lora/$a/run/progress.json
  echo "== $a"
  [ -f "$f" ] || { echo "   not started"; continue; }
  "$(grep -oP '^VENV=\K\S+' reasoning/hpc/hpc.conf | sed "s#\$ROOT#/scratch/$USER/bn4601a#")/bin/python" - "$f" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(f"   now: stage {p['chunk'] + 1}, {p['rows_done']:,} rows into it | {p.get('last_note', '')}")
print(f"   base MedQA {p.get('base_medqa_acc')} (direct {p.get('base_medqa_direct', '-')})")
for h in p["history"]:
    d = f" (direct {h['medqa_direct']:.1%})" if "medqa_direct" in h else ""
    print(f"   stage {h['stage']:>2}: seen {h['examples_seen']:>7,} | val_loss {h['val_loss']:.3f} | MedQA {h['medqa_acc']:.1%}{d} | {h['minutes']} min")
PY
done
ls reasoning/outputs/logs reasoning/outputs/metrics reasoning/outputs/embeddings 2>/dev/null
