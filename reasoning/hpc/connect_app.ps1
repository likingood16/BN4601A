# Open the vanda app (real AI) in Chrome on this laptop, through an SSH tunnel. Keep this window open while testing;
# close it (Ctrl-C) to disconnect. Run after app_on_vanda.ps1, once the app job shows R in qstat.
#   powershell -ExecutionPolicy Bypass -File reasoning\hpc\connect_app.ps1
$hostName = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"
$node = (ssh $hostName "cat $remote/reasoning/outputs/logs/app_server_host.txt 2>/dev/null").Trim()
if (-not $node) {
    Write-Host "The app job hasn't started yet (still queued). Check with: ssh nus-vanda qstat -u e0959158" -ForegroundColor Yellow
    exit 1
}
Write-Host "App job runs on $node. Loading MedGemma takes 2-4 minutes after the job starts."
Write-Host "Opening http://127.0.0.1:8010 in Chrome; refresh it if it says it can't connect yet."
Start-Process chrome "http://127.0.0.1:8010/#/"
Start-Process chrome "http://127.0.0.1:8010/#/board"
ssh -N -L "8010:${node}:8010" $hostName
