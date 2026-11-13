# Put the ED app with the REAL AI on vanda (6 Nov 2026). Run from the BN4601A folder, with the NUS VPN connected:
#   powershell -ExecutionPolicy Bypass -File reasoning\hpc\app_on_vanda.ps1
# 1. uploads the app (back end + built screens) and the FracAtlas train/val pictures the image upgrade jobs need
# 2. installs the web libraries if missing (FastAPI, Uvicorn, multipart), pinned by the constraints file
# 3. holds the long queued jobs so the app gets a GPU now (release later: ssh nus-vanda "qrls 1526195 1526342 1526343")
# 4. queues the app job, and copies one example picture of each kind to app/tests/test_images for testing
# Then run reasoning\hpc\connect_app.ps1 to open it in Chrome.
$ErrorActionPreference = "Stop"
$repo   = Resolve-Path "$PSScriptRoot\..\.."
$hostName = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"
$pack   = Join-Path $env:TEMP "app_pack.tar"

Write-Host "Packing ..."
tar -cf $pack -C $repo --exclude=app/frontend/node_modules --exclude=app/case_files --exclude=app/build `
    --exclude=app/frontend/src reasoning/src reasoning/checks reasoning/hpc app `
    vision/detector/train_detector.py vision/detector/choose_threshold.py vision/detector/finetune_medsam.py `
    vision/detector/data/yolo/images/train vision/detector/data/yolo/images/val `
    vision/detector/data/yolo/labels/train vision/detector/data/yolo/labels/val
Write-Host "Uploading (about 150 MB) ..."
scp -q $pack "${hostName}:$remote/app_pack.tar"
Remove-Item $pack

$cmd = @(
  "cd $remote && tar -xf app_pack.tar && rm app_pack.tar",
  "find reasoning/hpc -name '*.sh' -o -name '*.pbs' -o -name '*.conf' | xargs sed -i 's/\r$//'",
  "source reasoning/hpc/hpc.conf",
  "(`$VENV/bin/python -c 'import fastapi, uvicorn, multipart' 2>/dev/null || `$VENV/bin/pip install -q -c `$ROOT/constraints.txt fastapi uvicorn python-multipart)",
  "(qhold 1526195 1526342 1526343 2>/dev/null || true)",
  "rm -f reasoning/outputs/logs/app_server_host.txt",
  "qsub -q `$GPU_QUEUE -l `$GPU_SELECT -l walltime=10:00:00 -o reasoning/outputs/logs reasoning/hpc/jobs/app_server.pbs",
  "`$VENV/bin/python reasoning/hpc/pick_test_images.py",
  "qstat -u `$USER"
) -join " && "
ssh $hostName $cmd

Write-Host "Copying one example picture of each kind to app\tests\test_images ..."
New-Item -ItemType Directory -Force "$repo\app\tests\test_images" | Out-Null
scp -q "${hostName}:$remote/reasoning/data/image_tests/demo/*" "$repo\app\tests\test_images\"
Get-ChildItem "$repo\app\tests\test_images" | Select-Object Name, Length
Write-Host "Done. When the app job is running (R in qstat), run reasoning\hpc\connect_app.ps1" -ForegroundColor Green
