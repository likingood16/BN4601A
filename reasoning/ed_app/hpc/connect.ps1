# Open ED app v2 (real AI on vanda) in Chrome through an SSH tunnel. Keep this window open; Ctrl-C disconnects.
#   powershell -ExecutionPolicy Bypass -File reasoning\ed_app\hpc\connect.ps1
$hostName = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"
$node = (ssh $hostName "cat $remote/reasoning/outputs/logs/ed_app_host.txt 2>/dev/null").Trim()
if (-not $node) { Write-Host "The app job hasn't started yet. Check: ssh nus-vanda qstat -u e0959158" -ForegroundColor Yellow; exit 1 }
Write-Host "App runs on $node (MedGemma takes 2-4 min to load after the job starts)."
Start-Process chrome "http://127.0.0.1:8011/#/"
Start-Process chrome "http://127.0.0.1:8011/#/board"
ssh -N -L "8011:${node}:8011" $hostName
