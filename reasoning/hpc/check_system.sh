#!/bin/bash
# Run ON THE HPC LOGIN NODE (no GPU needed): reports what this account can use, so hpc.conf can be set right.
#   bash hpc/check_system.sh            (from the project folder on the HPC)
echo "== who / where";   whoami; hostname; cat /etc/os-release 2>/dev/null | grep PRETTY
echo "== python";        for p in python3 python3.11 python3.12; do command -v $p >/dev/null && echo "$p: $($p --version 2>&1)"; done
echo "== modules (python, cuda, singularity/apptainer, miniforge)"
module avail 2>&1 | tr ' ' '\n' | grep -iE "python|cuda|singularity|apptainer|conda|miniforge" | sort -u | head -40
echo "== container tools"; command -v singularity apptainer 2>/dev/null
echo "== job queues";    qstat -Q 2>/dev/null || qstat -q 2>/dev/null || echo "qstat not found"
echo "== GPU queues (details)"
for q in $(qstat -Q 2>/dev/null | awk 'NR>2{print $1}' | grep -iE "gpu|volta|hopper|a100|h100"); do
  echo "-- $q"; qstat -Qf $q 2>/dev/null | grep -iE "resources_max|resources_default|acl|enabled|started" ; done
echo "== storage";       for d in /home/svu/$USER /hpctmp/$USER /scratch/$USER /scratch2/$USER; do [ -d "$d" ] && df -h "$d" | tail -1 | sed "s|^|$d  |"; done
hpc s 2>/dev/null | head -20
echo "== internet from login node"; (curl -sI --max-time 8 https://huggingface.co >/dev/null && echo "huggingface.co reachable") || echo "NO internet on login node"
echo "== GPU on this node"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "no GPU on login node (normal)"
