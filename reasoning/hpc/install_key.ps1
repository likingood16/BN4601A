# One-time: copy this laptop's public SSH key to your NUS HPC account.
# You type your NUSNET password once; afterwards key login works without it.
$pub = (Get-Content "$env:USERPROFILE\.ssh\nus_hpc_ed25519.pub" -Raw).Trim()
Write-Host ""
Write-Host "Installing your laptop's SSH key on NUS HPC (vanda.nus.edu.sg)."
Write-Host "Type your NUSNET password when asked. Nothing shows while you type - that's normal." -ForegroundColor Yellow
Write-Host ""
ssh -o StrictHostKeyChecking=accept-new e0959158@vanda.nus.edu.sg "mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qxF '$pub' ~/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED"
Write-Host ""
Write-Host "Done. You can close this tab." -ForegroundColor Green
