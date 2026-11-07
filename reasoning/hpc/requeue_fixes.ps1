# 6 Nov 2026: upload the fixed code to vanda, then re-queue the two jobs that failed.
#   chest check (image_tools job 1525769): torchxrayvision clashed with our src/model.py, and the blur step failed.
#   pipeline trial (job 1525665): hpc.conf had Windows line endings (\r).
# Run from the repo root:  powershell -File reasoning\hpc\requeue_fixes.ps1
$ErrorActionPreference = "Stop"
$hostName = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"
Set-Location (Join-Path $PSScriptRoot "..\..")
$src = "reasoning/src"
scp -q "$src/image_tools.py" "$src/image_reader.py" "$src/pipeline.py" "$src/visit.py" "$src/charting.py" "${hostName}:$remote/reasoning/src/"
scp -q "reasoning/hpc/jobs/image_tools.pbs" "${hostName}:$remote/reasoning/hpc/jobs/"
$cmd = "cd $remote && find reasoning/hpc -name '*.sh' -o -name '*.pbs' -o -name '*.conf' | xargs sed -i 's/\r$//' && " +
       "source reasoning/hpc/hpc.conf && " +
       "qsub -q `$GPU_QUEUE -l `$GPU_SELECT -l walltime=02:00:00 -o reasoning/outputs/logs -v PARTS='chest sorter' reasoning/hpc/jobs/image_tools.pbs && " +
       "bash reasoning/hpc/start_trial.sh"
ssh $hostName $cmd
