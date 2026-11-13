# Upload ED app v2 to vanda and start (or restart) its job. Run from anywhere, with the NUS VPN connected:
#   powershell -ExecutionPolicy Bypass -File reasoning\ed_app\hpc\upload.ps1
# Builds the screens first if Node is installed. If the app job is already running it restarts with the new code
# (no queueing); otherwise it queues ed_app.pbs.
# It uploads ONLY the app (reasoning/ed_app + test files). The model code (reasoning/src, heads, adapters) on vanda is
# left as it is, so the app always runs on whatever models are there.
$ErrorActionPreference = "Stop"
$repo   = Resolve-Path "$PSScriptRoot\..\..\.."
$hostName = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"
$node = "$env:LOCALAPPDATA\Programs\node-v26.13.0-win-x64"
if (Test-Path "$node\npm.cmd") {
    Write-Host "Building the screens ..."
    $env:PATH = "$node;$env:PATH"
    Push-Location "$repo\reasoning\ed_app\ui"; & "$node\npm.cmd" run build | Out-Null; Pop-Location
}
$pack = Join-Path $env:TEMP "ed_app_pack.tar"
tar -cf $pack -C $repo --exclude=reasoning/ed_app/ui/node_modules --exclude=reasoning/ed_app/data `
    --exclude=__pycache__ reasoning/ed_app reasoning/app/test_backend.py `
    reasoning/web/test_notes reasoning/web/test_images
scp -q $pack "${hostName}:$remote/ed_app_pack.tar"
Remove-Item $pack
$cmd = @(
  "cd $remote && tar -xf ed_app_pack.tar && rm ed_app_pack.tar",
  "sed -i 's/\r$//' reasoning/ed_app/hpc/ed_app.pbs",
  "source reasoning/hpc/hpc.conf",
  ("if qstat -u `$USER | grep -q ed_app_v2; then touch reasoning/outputs/logs/ed_restart; echo 'restarting the running job'; " +
   "else qsub -q `$GPU_QUEUE -l `$GPU_SELECT -l walltime=10:00:00 -o reasoning/outputs/logs reasoning/ed_app/hpc/ed_app.pbs; fi"),
  "qstat -u `$USER"
) -join " && "
ssh $hostName $cmd
Write-Host "Then: powershell -ExecutionPolicy Bypass -File reasoning\ed_app\hpc\connect.ps1" -ForegroundColor Green
