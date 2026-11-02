# Laptop -> HPC: copy the code, the prepared data and the Colab LoRA checkpoint (about 600 MB, first time only).
# Needs: NUS Wi-Fi or NUS VPN, and the SSH key installed (install_key.ps1).
#   powershell -ExecutionPolicy Bypass -File reasoning\hpc\sync_up.ps1          (from the BN4601A folder)
$ErrorActionPreference = "Stop"
$repo   = Resolve-Path "$PSScriptRoot\..\.."
$host_  = "nus-vanda"
$remote = "/scratch/e0959158/bn4601a/BN4601A"

Write-Host "Creating folders on the HPC ..."
ssh $host_ "mkdir -p $remote/reasoning/data/processed $remote/reasoning/models/lora/mixed $remote/reasoning/outputs/logs"

Write-Host "Code (src, hpc) ..."
scp -r -q "$repo\reasoning\src" "$repo\reasoning\hpc" "${host_}:$remote/reasoning/"

Write-Host "Data (Pool 2, MedQA bench, patient cards) - the big file takes a few minutes ..."
$data = "pool2_train.jsonl", "pool2_val.jsonl", "pool2_test.jsonl", "medqa_bench200.jsonl",
        "pool1.parquet", "cards_pool1.parquet", "cards_multicare.parquet"
foreach ($f in $data) {
    Write-Host "  $f"
    scp -q "$repo\reasoning\data\processed\$f" "${host_}:$remote/reasoning/data/processed/"
}

Write-Host "Colab LoRA checkpoint (stage 1, 22,448 examples) ..."
scp -r -q "$repo\reasoning\models\lora\mixed\stage1_22448" "${host_}:$remote/reasoning/models/lora/mixed/"

Write-Host "Fixing Windows line endings in the shell scripts ..."
ssh $host_ "find $remote/reasoning/hpc -name '*.sh' -o -name '*.pbs' -o -name '*.conf' | xargs sed -i 's/\r$//'"

Write-Host "Done. Files on the HPC:" -ForegroundColor Green
ssh $host_ "du -sh $remote/reasoning/* 2>/dev/null"
