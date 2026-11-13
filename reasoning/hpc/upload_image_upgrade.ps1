# Upload the image-upgrade code to vanda and queue it (6 Nov 2026).  powershell -File reasoning\hpc\upload_image_upgrade.ps1
$ErrorActionPreference = "Stop"
$hostName = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"
Set-Location (Join-Path $PSScriptRoot "..\..")
scp -q vision/detector/train_detector.py vision/detector/choose_threshold.py vision/detector/finetune_medsam.py "${hostName}:$remote/vision/detector/"
scp -q reasoning/hpc/jobs/image_upgrade.pbs "${hostName}:$remote/reasoning/hpc/jobs/"
scp -q reasoning/hpc/start_image_upgrade.sh "${hostName}:$remote/reasoning/hpc/"
ssh $hostName "cd $remote && sed -i 's/\r$//' reasoning/hpc/start_image_upgrade.sh reasoning/hpc/jobs/image_upgrade.pbs reasoning/hpc/hpc.conf && bash reasoning/hpc/start_image_upgrade.sh"
