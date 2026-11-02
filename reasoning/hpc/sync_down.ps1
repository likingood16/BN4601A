# HPC -> laptop: bring back LoRA checkpoints, metrics, logs and (small) embeddings into the same folders.
#   powershell -ExecutionPolicy Bypass -File reasoning\hpc\sync_down.ps1        (from the BN4601A folder)
$ErrorActionPreference = "Continue"
$repo   = Resolve-Path "$PSScriptRoot\..\.."
$host_  = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A/reasoning"

New-Item -ItemType Directory -Force "$repo\reasoning\models\lora", "$repo\reasoning\outputs" | Out-Null
Write-Host "LoRA checkpoints (each ~120 MB) ..."
foreach ($arm in "mixed", "emergency", "two_stage") {
    ssh $host_ "test -d $remote/models/lora/$arm" 2>$null
    # no "\" at the end of a destination: Windows PowerShell turns \" into a stray quote and scp fails
    if ($LASTEXITCODE -eq 0) { scp -r -q "${host_}:$remote/models/lora/$arm" "$repo\reasoning\models\lora" ; Write-Host "  $arm" }
}
Write-Host "Score heads, metrics, logs, embeddings ..."
scp -r -q "${host_}:$remote/models/heads" "$repo\reasoning\models"
foreach ($d in "metrics", "logs", "embeddings") { scp -r -q "${host_}:$remote/outputs/$d" "$repo\reasoning\outputs" }
Write-Host "Done." -ForegroundColor Green
